import cv2
import numpy as np
import zxingcpp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os


class Code128Hunter:
    """
    Code128 条码复杂场景识别引擎
    核心设计思想：
      1. 分层识别策略：从最快→最慢、最简→最复杂，命中即停止，节省算力
      2. 全流程性能优化：向量化、惰性求值、缓冲区复用、预计算LUT、形态学核
      3. 多线程并发解码：全局线程池复用，避免频繁创建销毁
      4. 低分辨率保护：<200万像素不缩放，保证小图识别率
      5. 区域定位优先：梯度/扫描线定位ROI，避免全图暴力搜索

    识别流程（6层，按速度/成本排序）：
      Layer0 → 全图直接解码（最快）
      Layer1 → 全图多预处理增强解码
      Layer2 → 多尺度缩放尝试
      Layer3 → 梯度边缘定位ROI（精准定位）
      Layer4 → 扫描线特征检测定位条码区域
      Layer5 → 暴力全角度旋转（兜底策略）
    """

    # 工作图默认缩放系数（大图降采样提升速度）
    WORK_SCALE = 0.25

    def __init__(self, max_workers=None):
        """
        初始化识别引擎
        :param max_workers: 线程池最大线程数，默认取CPU核心数与8的较小值
        """
        # Code128 最小模块数（用于过滤无效区域）
        self.min_modules = 46
        # 线程安全锁：保护识别结果多线程竞争
        self._lock = threading.Lock()
        # 事件标志：一旦找到条码，立即停止所有任务
        self._found = threading.Event()
        # 存储最终识别结果
        self._result = None
        # 配置线程数：自动适配CPU核心，避免超线程浪费
        self.max_workers = max_workers or min(8, (os.cpu_count() or 4))
        # 全局复用线程池：避免反复创建/销毁线程带来的性能损耗
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # ===================== 预计算资源（启动时一次计算，全程复用）=====================
        # 预计算 Gamma 增强 LUT 查找表：加速图像亮度调整，避免实时幂运算
        self._gamma_luts = {}
        for gamma in [0.5, 0.7, 1.5, 2.0]:
            table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in range(256)], dtype=np.uint8)
            self._gamma_luts[gamma] = table

        # 预计算锐化卷积核：增强条码边缘
        self._kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)

        # 预计算形态学核：专门用于条码区域定位（不同宽高比适配横/竖/斜条码）
        self._morph_kernels_locate = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 21)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25)),
        ]
        # 形态学开运算/闭运算核：去噪、连接断条
        self._morph_k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        self._morph_k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

    def __del__(self):
        """析构函数：关闭线程池，释放资源"""
        self._executor.shutdown(wait=False)

    def hunt(self, image_path):
        """
        【主入口】对外暴露的条码识别方法
        :param image_path: 图片路径
        :return: 识别结果列表 / None
        """
        # 读取灰度图：直接灰度读取，省去彩色转灰度步骤
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"读不到: {image_path}")
            return None

        h, w = img.shape[:2]
        print(f"原图尺寸: {w}x{h}")

        gray_full = img  # 原图灰度（全分辨率，用于最终精识别）

        # ===================== 构建工作图（降采样策略）=====================
        total_pixels = h * w

        # 低分辨率保护：<200万像素不缩放，避免分辨率过低无法识别
        if total_pixels < 2_000_000:
            print(f"总像素 {total_pixels / 1e6:.1f}M < 2M，跳过缩放，直接使用原图")
            self.scale = 1.0
            gray_work = gray_full
        else:
            # 大图降采样：加速定位与预处理
            self.scale = self.WORK_SCALE
            sw, sh = int(w * self.scale), int(h * self.scale)
            # 保证工作图最小尺寸，避免过小失效
            if sw < 200 or sh < 100:
                self.scale = max(200.0 / w, 100.0 / h, self.WORK_SCALE)
                sw, sh = int(w * self.scale), int(h * self.scale)
            if self.scale >= 1.0:
                self.scale = 1.0
                gray_work = gray_full
            else:
                # 区域插值降采样：保留更多结构信息
                gray_work = cv2.resize(gray_full, (sw, sh), interpolation=cv2.INTER_AREA)

        print(f"工作图尺寸: {gray_work.shape[1]}x{gray_work.shape[0]} (scale={self.scale:.2f})")

        # 缩放反比：用于坐标映射（工作图→原图）
        self._inv_scale = 1.0 / self.scale

        # 重置状态：每次识别前清空结果
        self._found.clear()
        self._result = None

        # ===================== 6层策略依次尝试，找到即返回 =====================
        for layer_func in [
            lambda: self._layer0_direct(gray_work, gray_full),
            lambda: self._layer1_multi_preprocess(gray_work, gray_full),
            lambda: self._layer2_multi_scale(gray_work, gray_full),
            lambda: self._layer3_gradient_locate(gray_work, gray_full),
            lambda: self._layer4_scanline_locate(gray_work, gray_full),
            lambda: self._layer5_brute_rotate(gray_work, gray_full),
        ]:
            result = layer_func()
            if result:
                return result

        print("✗ 所有方法均未找到")
        return None

    # ========================================================
    #  解码核心 - 内联检查found减少函数调用开销
    # ========================================================
    def _try_decode(self, gray, tag=""):
        """
        单张图像条码解码（线程安全、命中即停）
        :param gray: 灰度图
        :param tag: 调试标记，用于打印策略来源
        :return: 结果 / None
        """
        # 已找到：直接退出，节省算力
        if self._found.is_set():
            return None
        try:
            # 只解码 Code128，开启自动旋转/降采样，提高兼容性
            results = zxingcpp.read_barcodes(
                gray,
                formats=zxingcpp.BarcodeFormat.Code128,
                try_rotate=True,
                try_downscale=True,
            )
            if results:
                # 加锁确保只有第一个结果被保存
                with self._lock:
                    if not self._found.is_set():
                        self._found.set()
                        self._result = results
                        for r in results:
                            print(f"  ✓ [{tag}] type={r.format} data=\"{r.text}\"")
                        return results
        except Exception:
            pass
        return None

    def _try_decode_batch(self, image_tag_pairs):
        """
        批量图像并发解码
        :param image_tag_pairs: [(图像, 标记), ...]
        :return: 第一个成功的结果
        """
        if not image_tag_pairs:
            return None

        # 少数量任务串行更快：避免线程调度开销
        if len(image_tag_pairs) <= 2:
            for img, tag in image_tag_pairs:
                if self._found.is_set():
                    return self._result
                result = self._try_decode(img, tag)
                if result:
                    return result
            return None

        # 多任务提交线程池
        futures = {}
        for img, tag in image_tag_pairs:
            if self._found.is_set():
                break
            f = self._executor.submit(self._try_decode, img, tag)
            futures[f] = tag

        # 等待任一任务完成，成功则取消所有剩余任务
        for f in as_completed(futures):
            result = f.result()
            if result:
                for remaining in futures:
                    remaining.cancel()
                return result

        return self._result

    # ========================================================
    #  裁剪旋转 - 合并为单一高效函数
    #  功能：从原图裁剪感兴趣区域并旋转矫正，用于ROI精识别
    # ========================================================
    def _crop_and_rotate_full(self, gray_full, cx, cy, size, angle):
        h, w = gray_full.shape
        half = size >> 1  # 位运算替代整除，速度更快
        # 边界保护，防止越界
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)

        # 区域过小无效，直接跳过
        if y2 - y1 < 20 or x2 - x1 < 20:
            return None

        crop = gray_full[y1:y2, x1:x2]

        # 角度很小，不需要旋转
        if abs(angle) < 0.5:
            return crop

        # 旋转并扩展画布，避免内容被截断
        ch, cw = crop.shape
        rc = (cw >> 1, ch >> 1)
        M = cv2.getRotationMatrix2D(rc, angle, 1.0)
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        nw = int(ch * sin_a + cw * cos_a)
        nh = int(ch * cos_a + cw * sin_a)
        M[0, 2] += (nw - cw) * 0.5
        M[1, 2] += (nh - ch) * 0.5
        return cv2.warpAffine(crop, M, (nw, nh),
                              flags=cv2.INTER_LINEAR, borderValue=255)

    # ========================================================
    #  预处理生成器 - 惰性求值 + 每步检查found
    #  特点：不一次性生成所有图，用一个生成一个，节省内存
    # ========================================================
    def _gen_preprocessed(self, gray):
        """惰性生成，每次yield前检查是否已找到"""
        if self._found.is_set(): return

        # 对比度受限自适应直方图均衡化
        clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yield ("CLAHE_2", clahe2.apply(gray))
        if self._found.is_set(): return

        # 大津阈值二值化
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield ("Otsu", otsu)
        if self._found.is_set(): return

        # 自适应高斯阈值
        yield ("AdaptGauss",
               cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 10))
        if self._found.is_set(): return

        # 高斯模糊去噪
        yield ("Gauss3", cv2.GaussianBlur(gray, (3, 3), 0))
        if self._found.is_set(): return

        # 锐化增强
        yield ("Sharpen", cv2.filter2D(gray, -1, self._kernel_sharp))
        if self._found.is_set(): return

        # 全局直方图均衡化
        yield ("HistEq", cv2.equalizeHist(gray))
        if self._found.is_set(): return

        # 不同强度CLAHE
        for clip in [4.0, 8.0]:
            if self._found.is_set(): return
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            yield (f"CLAHE_{clip}", clahe.apply(gray))

        if self._found.is_set(): return
        # 自适应均值阈值
        yield ("AdaptMean",
               cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 15, 10))

        # Gamma亮度增强（使用预计算LUT）
        for gamma, lut in self._gamma_luts.items():
            if self._found.is_set(): return
            yield (f"Gamma_{gamma}", cv2.LUT(gray, lut))

        if self._found.is_set(): return
        # 颜色反转（适应白底黑条/黑底白条）
        yield ("Invert", cv2.bitwise_not(gray))

        if self._found.is_set(): return
        # 双边滤波（保边去噪）
        yield ("Bilateral", cv2.bilateralFilter(gray, 9, 75, 75))

        if self._found.is_set(): return
        yield ("Gauss5", cv2.GaussianBlur(gray, (5, 5), 0))

        if self._found.is_set(): return
        # 形态学开运算
        yield ("MorphOpen", cv2.morphologyEx(gray, cv2.MORPH_OPEN, self._morph_k_open))

        if self._found.is_set(): return
        # 形态学闭运算
        yield ("MorphClose", cv2.morphologyEx(gray, cv2.MORPH_CLOSE, self._morph_k_close))

    def _multi_preprocess_decode(self, gray, tag):
        """
        多预处理策略批量解码（核心增强方法）
        先尝试原图，再依次尝试所有增强图
        """
        if self._found.is_set():
            return self._result

        # 第一步：直接尝试原图
        result = self._try_decode(gray, f"{tag}/原图")
        if result:
            return result

        # 第二步：分批提交预处理图（每批=线程数，避免任务堆积）
        batch = []
        for name, img in self._gen_preprocessed(gray):
            if self._found.is_set():
                return self._result
            batch.append((img, f"{tag}/{name}"))
            if len(batch) >= self.max_workers:
                result = self._try_decode_batch(batch)
                if result:
                    return result
                batch.clear()

        # 处理剩余批次
        if batch:
            result = self._try_decode_batch(batch)
            if result:
                return result

        return None

    # ========================================================
    #  两阶段解码：工作图粗识别 → 原图精识别（速度+精度平衡）
    # ========================================================
    def _two_stage_decode(self, work_img, full_img, tag):
        result = self._try_decode(work_img, f"{tag}/low")
        if result:
            return result
        # 工作图没找到，且原图更大，则尝试原图
        if self.scale < 1.0 and full_img is not None:
            result = self._try_decode(full_img, f"{tag}/full")
            if result:
                return result
        return None

    def _two_stage_multi_preprocess(self, work_img, full_img, tag):
        """两阶段 + 多预处理增强（最常用高精度策略）"""
        result = self._multi_preprocess_decode(work_img, f"{tag}/low")
        if result:
            return result
        if self.scale < 1.0 and full_img is not None:
            result = self._multi_preprocess_decode(full_img, f"{tag}/full")
            if result:
                return result
        return None

    # ========================================================
    #  第0层：最快策略 - 全图直接解码（无任何增强）
    # ========================================================
    def _layer0_direct(self, gray_work, gray_full):
        return self._two_stage_decode(gray_work, gray_full, "全图直接")

    # ========================================================
    #  第1层：快速策略 - 全图多预处理增强
    # ========================================================
    def _layer1_multi_preprocess(self, gray_work, gray_full):
        return self._two_stage_multi_preprocess(gray_work, gray_full, "全图多预处理")

    # ========================================================
    #  第2层：多尺度策略 - 不同缩放大小尝试（适应大小条码）
    # ========================================================
    def _layer2_multi_scale(self, gray_work, gray_full):
        h, w = gray_work.shape
        scales = [0.5, 2.0, 0.75, 1.5, 3.0, 4.0]

        for s in scales:
            if self._found.is_set():
                return self._result

            nw, nh = int(w * s), int(h * s)
            # 过滤无效尺寸
            if nw < 100 or nh < 50 or nw > 4000 or nh > 4000:
                continue

            # 缩放插值方式：缩小用AREA（保结构），放大用LINEAR（速度快）
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
            scaled = cv2.resize(gray_work, (nw, nh), interpolation=interp)

            result = self._try_decode(scaled, f"多尺度/ws{s}")
            if result:
                return result

            result = self._multi_preprocess_decode(scaled, f"多尺度/ws{s}")
            if result:
                return result

        return None

    # ========================================================
    #  第3层：梯度定位（核心高精度定位）
    #  原理：利用条码高梯度、高对比度、条状结构定位区域
    # ========================================================
    def _layer3_gradient_locate(self, gray_work, gray_full):
        # 双尺度定位：1.0 + 0.5，提高小条码召回率
        for sub_scale in [1.0, 0.5]:
            if self._found.is_set():
                return self._result

            if sub_scale != 1.0:
                sh, sw = gray_work.shape
                work = cv2.resize(gray_work, (int(sw * sub_scale), int(sh * sub_scale)),
                                  interpolation=cv2.INTER_AREA)
                total_scale = self.scale * sub_scale
            else:
                work = gray_work
                total_scale = self.scale

            inv_total = 1.0 / total_scale
            # 执行梯度定位，获取候选ROI
            rois = self._find_barcode_regions(work)
            print(f"  梯度定位/s{total_scale:.2f}: 找到 {len(rois)} 个候选区域")

            if not rois:
                continue

            # 对每个候选区域进行裁剪旋转+精识别
            for i, (rect, angle) in enumerate(rois):
                if self._found.is_set():
                    return self._result

                cx_w, cy_w = rect[0]
                bw, bh = rect[1]

                # 映射到原图并裁剪旋转
                full_roi = self._crop_and_rotate_full(
                    gray_full, int(cx_w * inv_total), int(cy_w * inv_total),
                    int(max(bw, bh) * inv_total * 1.5), angle)

                # 工作图兜底
                if full_roi is None:
                    work_roi = self._rectify_roi(work, rect, angle)
                    if work_roi is None:
                        continue
                    full_roi = work_roi

                tag = f"梯度/s{total_scale:.2f}/ROI{i}"
                result = self._multi_preprocess_decode(full_roi, tag)
                if result:
                    return result

                # 翻转再尝试一次（兼容反向条码）
                flipped = cv2.flip(full_roi, -1)
                result = self._multi_preprocess_decode(flipped, f"{tag}_flip")
                if result:
                    return result

        return None

    def _find_barcode_regions(self, gray):
        """
        梯度分析定位条码区域
        步骤：梯度计算 → 边缘增强 → 形态学闭合 → 轮廓提取 → 条件过滤
        """
        # 高斯模糊降噪
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # Scharr 算子计算XY方向梯度（比Sobel更精准）
        gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)

        abs_gx = np.abs(gx)
        abs_gy = np.abs(gy)
        mag = cv2.magnitude(gx, gy)       # 梯度幅值
        angle_map = cv2.phase(gx, gy)     # 梯度方向

        # 条码特征：水平/垂直梯度差异明显
        diff1 = cv2.convertScaleAbs(abs_gx - abs_gy)
        mag_max = mag.max()
        if mag_max < 1e-6:
            return []
        mag_u8 = cv2.convertScaleAbs(mag * (255.0 / mag_max))

        candidates = []

        # 分别用梯度差与梯度幅值寻找候选区
        for gradient_map in [diff1, mag_u8]:
            blurred = cv2.GaussianBlur(gradient_map, (9, 9), 0)
            _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

            # 多形态学核适配不同方向条码
            for kernel in self._morph_kernels_locate:
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                closed = cv2.dilate(closed, None, iterations=3)
                closed = cv2.erode(closed, None, iterations=2)

                # 提取外轮廓
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < 500:    # 过滤太小区域
                        continue
                    rect = cv2.minAreaRect(c)
                    bw, bh = rect[1]
                    if bw < 1 or bh < 1:
                        continue
                    aspect = max(bw, bh) / min(bw, bh)
                    # 条码必须是长条型：过滤宽高比不合理区域
                    if aspect < 1.5 or aspect > 30:
                        continue

                    # 精算角度
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    precise_angle = self._estimate_angle(angle_map, mag, mask)
                    candidates.append((rect, precise_angle))

        # 非极大值抑制：去掉重叠框
        return self._nms_rects(candidates)

    def _estimate_angle(self, angle_map, mag, mask):
        """加权梯度方向直方图：估算条码倾斜角"""
        valid_mask = mask > 0
        mag_vals = mag[valid_mask]
        n = mag_vals.size
        if n < 10:
            return 0

        # 只取高幅值点计算角度，更精准
        mid = n >> 1
        threshold = np.partition(mag_vals.ravel(), -mid)[-mid]
        strong = valid_mask & (mag > threshold)
        strong_count = strong.sum()
        if strong_count < 10:
            return 0

        angles = angle_map[strong]
        weights = mag[strong]
        angles_mod = np.mod(angles, np.pi)
        # 加权直方图统计主方向
        hist, _ = np.histogram(angles_mod, bins=180, range=(0, np.pi), weights=weights)
        hist_smooth = cv2.GaussianBlur(hist.astype(np.float32).reshape(1, -1),
                                       (1, 15), 0).flatten()
        return int(np.argmax(hist_smooth)) - 90

    def _nms_rects(self, candidates, dist_ratio=0.5):
        """非极大值抑制(NMS)：保留面积最大、去掉重叠框"""
        if len(candidates) <= 1:
            return candidates
        # 按面积从大到小排序
        candidates.sort(key=lambda x: x[0][1][0] * x[0][1][1], reverse=True)
        centers = np.array([c[0][0] for c in candidates])
        sizes = np.array([max(c[0][1]) for c in candidates])
        keep = []
        suppressed = set()
        for i in range(len(candidates)):
            if i in suppressed:
                continue
            keep.append(candidates[i])
            # 距离过近的框抑制
            if i + 1 < len(candidates):
                dists = np.linalg.norm(centers[i + 1:] - centers[i], axis=1)
                for j_off in np.where(dists < sizes[i] * dist_ratio)[0]:
                    suppressed.add(i + 1 + j_off)
        return keep

    def _rectify_roi(self, gray, rect, angle):
        """从工作图中裁剪并旋转ROI（备用方法）"""
        center_x, center_y = int(rect[0][0]), int(rect[0][1])
        bw, bh = int(rect[1][0]), int(rect[1][1])
        crop_size = int(max(bw, bh) * 1.3)
        half = crop_size >> 1
        h, w = gray.shape
        y1, y2 = max(0, center_y - half), min(h, center_y + half)
        x1, x2 = max(0, center_x - half), min(w, center_x + half)
        if y2 - y1 < 20 or x2 - x1 < 20:
            return None
        crop = gray[y1:y2, x1:x2]
        if abs(angle) < 0.5:
            return crop
        ch, cw = crop.shape
        rc = (cw >> 1, ch >> 1)
        M = cv2.getRotationMatrix2D(rc, angle, 1.0)
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        nw = int(ch * sin_a + cw * cos_a)
        nh = int(ch * cos_a + cw * sin_a)
        M[0, 2] += (nw - cw) * 0.5
        M[1, 2] += (nh - ch) * 0.5
        return cv2.warpAffine(crop, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=255)

    # ========================================================
    #  第4层：扫描线定位（向量化优化，极快）
    #  原理：条码具有周期性跳变特征，用扫描线检测跳变规律
    # ========================================================
    def _layer4_scanline_locate(self, gray_work, gray_full):
        wh, ww = gray_work.shape
        work_blur = cv2.GaussianBlur(gray_work, (3, 3), 0)
        inv = self._inv_scale

        hit_crops = []
        batch_limit = self.max_workers * 2

        # 预计算角度向量（向量化加速）
        angles = np.arange(0, 180, 5)
        rads = np.radians(angles)
        dxs = np.cos(rads)
        dys = np.sin(rads)

        num_lines = 30
        max_dim = max(ww, wh)
        max_steps = min(int(max_dim * 1.5), 2000)
        steps = np.arange(max_steps, dtype=np.float32)

        for ai, angle in enumerate(angles):
            if self._found.is_set():
                return self._result

            dx, dy = dxs[ai], dys[ai]

            # 多线扫描，覆盖全图
            for li in range(num_lines):
                # 按角度方向生成扫描起点
                if abs(dx) > abs(dy):
                    sx, sy = 0.0, wh * (li + 0.5) / num_lines
                else:
                    sx, sy = ww * (li + 0.5) / num_lines, 0.0

                # 向量化计算整条扫描线像素坐标
                xs = (sx + steps * dx).astype(np.int32)
                ys = (sy + steps * dy).astype(np.int32)

                # 边界检查（向量化，无循环）
                valid = (xs >= 0) & (xs < ww) & (ys >= 0) & (ys < wh)
                invalid_idx = np.argmin(valid)
                if invalid_idx == 0:
                    if not valid[0]:
                        continue
                    n_valid = len(valid)
                else:
                    n_valid = invalid_idx

                if n_valid < 50:
                    continue

                # 提取有效线段像素
                xs_v = xs[:n_valid]
                ys_v = ys[:n_valid]
                pixels = work_blur[ys_v, xs_v].astype(np.float32)

                # 判断是否为条码特征线段
                if self._is_barcode_scanline_fast(pixels):
                    # 计算中心点与角度
                    cx_w = sx + dx * n_valid * 0.5
                    cy_w = sy + dy * n_valid * 0.5
                    barcode_angle = angle - 90
                    size_w = n_valid * 1.5

                    # 映射到原图裁剪
                    full_roi = self._crop_and_rotate_full(
                        gray_full, int(cx_w * inv), int(cy_w * inv),
                        int(size_w * inv), barcode_angle)

                    if full_roi is not None:
                        hit_crops.append((full_roi, f"扫描线/a{angle}/L{li}"))

                    # 批次满则提交解码
                    if len(hit_crops) >= batch_limit:
                        result = self._try_decode_batch(hit_crops)
                        if result:
                            return result
                        hit_crops.clear()

        # 处理剩余命中区域
        if hit_crops:
            result = self._try_decode_batch(hit_crops)
            if result:
                return result

            for img, tag in hit_crops:
                if self._found.is_set():
                    return self._result
                result = self._multi_preprocess_decode(img, tag)
                if result:
                    return result

        return None

    def _is_barcode_scanline_fast(self, pixels):
        """快速判断扫描线是否符合条码跳变特征（无循环，全向量化）"""
        n = len(pixels)
        pmin, pmax = pixels.min(), pixels.max()

        # 条件1：对比度必须足够
        if pmax - pmin < 30:
            return False

        # 条件2：黑白跳变次数足够多
        med = (pmax + pmin) * 0.5
        binary = (pixels > med)
        diff = np.diff(binary)
        transitions = np.count_nonzero(diff)

        if transitions < 20:
            return False

        # 条件3：跳变密度在合理范围
        density = transitions / n
        if density < 0.05 or density > 0.5:
            return False

        # 条件4：滑动窗口验证周期性
        if n >= 50:
            diff_u8 = np.abs(diff).astype(np.uint8)
            cs = np.cumsum(diff_u8)
            cs = np.insert(cs, 0, 0)
            if len(cs) > 50:
                window_sums = cs[50:] - cs[:-50]
                if window_sums.max() < 10:
                    return False
            elif cs[-1] < 10:
                return False
        elif transitions < 10:
            return False

        return True

    # ========================================================
    #  第5层：暴力旋转兜底（全角度搜索，最慢但最稳）
    # ========================================================
    def _layer5_brute_rotate(self, gray_work, gray_full):
        wh, ww = gray_work.shape
        center = (ww >> 1, wh >> 1)

        # 第一轮：工作图批量旋转（快速兜底）
        batch = []
        batch_size = self.max_workers * 4

        for angle in range(5, 360, 5):
            if self._found.is_set():
                return self._result

            rad = np.radians(angle)
            cos_a, sin_a = abs(np.cos(rad)), abs(np.sin(rad))
            nw = int(wh * sin_a + ww * cos_a)
            nh = int(wh * cos_a + ww * sin_a)

            # 旋转并扩展画布
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += (nw - ww) * 0.5
            M[1, 2] += (nh - wh) * 0.5
            rotated = cv2.warpAffine(gray_work, M, (nw, nh),
                                     flags=cv2.INTER_LINEAR, borderValue=255)
            batch.append((rotated, f"暴力旋转/{angle}°/low"))

            # 批量提交
            if len(batch) >= batch_size:
                result = self._try_decode_batch(batch)
                if result:
                    return result
                batch.clear()

        if batch:
            result = self._try_decode_batch(batch)
            if result:
                return result

        # 第二轮：原图大角度步长 + 多预处理（终极兜底）
        fh, fw = gray_full.shape
        f_center = (fw >> 1, fh >> 1)

        for angle in range(5, 360, 15):
            if self._found.is_set():
                return self._result

            rad = np.radians(angle)
            cos_a, sin_a = abs(np.cos(rad)), abs(np.sin(rad))
            nw = int(fh * sin_a + fw * cos_a)
            nh = int(fh * cos_a + fw * sin_a)

            M = cv2.getRotationMatrix2D(f_center, angle, 1.0)
            M[0, 2] += (nw - fw) * 0.5
            M[1, 2] += (nh - fh) * 0.5
            rotated = cv2.warpAffine(gray_full, M, (nw, nh),
                                     flags=cv2.INTER_LINEAR, borderValue=255)

            result = self._multi_preprocess_decode(rotated, f"暴力旋转/{angle}°/full")
            if result:
                return result

        return None


# ========================================================
#  使用示例
# ========================================================
if __name__ == "__main__":
    hunter = Code128Hunter()
    hunter.hunt("test.jpg")
