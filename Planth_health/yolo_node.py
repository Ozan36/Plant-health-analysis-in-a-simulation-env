import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
# İmage_processing modülünden gerekli fonksiyonları dahil etme 
from image_processing import (
    apply_band_filter, 
    create_simple_green_mask, 
    clean_mask, 
    apply_mask_to_image
)

try:
    from sensor_msgs.msg import Image as ROSImage
    from cv_bridge import CvBridge
    from rclpy.node import Node
    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False
    Node = object


class YOLOv8MaskNode(Node if ROS_AVAILABLE else object):
    """YOLOv8 mask detection node"""
    
    def __init__(self):
        # ROS başlatma ve Bridge hazırlığı
        if ROS_AVAILABLE:
            super().__init__('yolov8_mask_node')
            self.bridge = CvBridge()
        else:
            self.get_logger = lambda: type('', (), {
                'info': print,
                'error': print,
                'warning': print
            })()
        
        # Görüntü depolama değişkenleri
        self.cv_image = None
        self.original_image = None
        self.mask_image = None
        self.last_mask = None
        self.last_corrected = None
        self.band_filtered_image = None
        
        # Thread güvenliği
        self.lock = threading.Lock()
        self.running = True
        
        # İşlem parametreleri(başlangıç parametresi)
        self.current_band = 112
        self.processing_time = 0.0
        self.detection_count = 0
        
        # Model ayarları(Bounding box çıkarma,maske kenar iyileştirme)
        self.verbose = False
        self.boxes = False
        self.overlap_mask = False
        self.retina_masks = True
        
        # YOLO model yükleme
        self._load_model()
        
        # ROS subscription veya simülasyon
        if ROS_AVAILABLE:
            self.create_subscription(
                ROSImage,
                '/a200_0000/sensors/camera_0/color/image',
                self.image_callback,
                10
            )
        else:
            threading.Thread(target=self._simulation_loop, daemon=True).start()
        
        # Çıkarım thread'i
        threading.Thread(target=self._inference_loop, daemon=True).start()
    
    def _load_model(self):
        """YOLO modelini yükler"""
        try:
            self.model = YOLO('models/best.pt')
            self.get_logger().info('Model yüklendi')
        except Exception as e:
            self.get_logger().error(f'Model yüklenemedi: {e}')
            self.model = None
    
    def _simulation_loop(self):
        """Test için simülasyon döngüsü"""
        while self.running:
            h, w = 480, 640
            img = np.random.randint(50, 150, (h, w, 3), dtype=np.uint8)
            
            # Rastgele yeşil elipsler (yaprak simülasyonu)
            for _ in range(18):
                cx = np.random.randint(60, w - 60)
                cy = np.random.randint(60, h - 60)
                sx = np.random.randint(30, 80)
                sy = np.random.randint(20, 60)
                ang = np.random.randint(0, 180)
                color = (0, np.random.randint(100, 200), 0)
                cv2.ellipse(img, (cx, cy), (sx, sy), ang, 0, 360, color, -1)
            
            with self.lock:
                self.cv_image = img.copy()
            
            time.sleep(0.5)
    
    def _run_model(self, img):
        """
        YOLO modelini çalıştırır
        
        Args:
            img: BGR görüntü
        
        Returns:
            (combined_mask, boxes) tuple
        """
        if not self.model:
            return None, None
        
        try:
            kwargs = dict(conf=0.45, iou=0.55, imgsz=640, device='cpu')
            
            try:
                res = self.model(
                    img, **kwargs,
                    verbose=self.verbose,
                    boxes=self.boxes,
                    overlap_mask=self.overlap_mask,
                    retina_masks=self.retina_masks
                )
            except TypeError:
                res = self.model(img, **kwargs)
            
            if res and len(res) > 0 and hasattr(res[0], 'masks') and res[0].masks is not None:
                masks = res[0].masks.data.cpu().numpy()
                
                if self.overlap_mask:
                    summed = np.sum(masks.astype(np.uint8), axis=0)
                    max_val = summed.max() if summed.max() > 0 else 1
                    combined = (np.clip(summed, 0, 5) / max_val * 255).astype(np.uint8)
                else:
                    combined = (np.any(masks, axis=0).astype(np.uint8) * 255)
                
                boxes = res[0].boxes if hasattr(res[0], 'boxes') else None
                return combined, boxes
        
        except Exception as e:
            if self.verbose:
                print('Model hata:', e)
        
        return None, None
    
    def _inference_loop(self):
        """Ana çıkarım döngüsü"""
        interval = 0.8
        
        while self.running:
            t0 = time.time()
            
            # Görüntüyü al
            with self.lock:
                src = self.cv_image.copy() if self.cv_image is not None else None
            
            if src is not None:
                corrected = src.copy()
                
                # Model ile maske oluştur
                mask, boxes = self._run_model(corrected)
                
                # Model başarısız olursa basit yeşil maske
                if mask is None:
                    mask = create_simple_green_mask(corrected)
                    boxes = None
                
                # Maske boyutunu kontrol et
                if mask is not None and mask.shape[:2] != corrected.shape[:2]:
                    mask = cv2.resize(
                        mask,
                        (corrected.shape[1], corrected.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                
                # Band filtresi uygula
                band_img = apply_band_filter(corrected, self.current_band, self.verbose)
                white = np.ones_like(band_img) * 255
                
                if mask is not None:
                    # Maskeyi temizle
                    clean = clean_mask(mask)
                    
                    # Maskeyi uygula
                    mask_img = apply_mask_to_image(band_img, clean)
                    
                    # Sonuçları kaydet
                    with self.lock:
                        self.last_mask = clean.copy()
                        self.last_corrected = corrected.copy()
                        self.mask_image = mask_img
                        self.original_image = corrected.copy()
                        self.detection_count = len(boxes) if (boxes is not None and self.boxes) else 0
                else:
                    with self.lock:
                        self.mask_image = white.copy()
                        self.original_image = corrected.copy()
                        self.detection_count = 0
                
                self.processing_time = time.time() - t0
                
                if self.verbose:
                    print(f'Çıkarım: {self.processing_time:.3f}s | '
                          f'Tespit: {self.detection_count} | '
                          f'Band: {self.current_band}')
            
            # Bekleme
            time.sleep(max(0.01, interval - (time.time() - t0)))
    
    def image_callback(self, msg):
        """ROS görüntü callback"""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.lock:
                self.cv_image = img.copy()
        except Exception as e:
            self.get_logger().error(f'Callback hatası: {e}')
    
    def stop(self):
        """Node'u durdur"""
        self.running = False

