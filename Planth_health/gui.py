#Gerekli kütüphanelerin eklenmesi
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image as PILImage
import cv2
import numpy as np
from image_processing import get_band_type, apply_band_filter, clean_mask, apply_mask_to_image

class TkinterGUI:
    """Tkinter tabanlı kullanıcı arayüzü"""
    
    def __init__(self, node):
 
        self.node = node
        self.root = tk.Tk()
        self.root.title("YOLOv8 Health - 224 Band Hyperspectral")
        self.slider_after = None
        
        self._build_ui()
        self.root.after(100, self._update_ui)
    
    def _build_ui(self):
        """UI elemanlarını oluşturur"""
        # Ana frame
        frm = ttk.Frame(self.root)
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Başlık
        ttk.Label(
            frm,
            text="YOLOv8 Health Detection - 224 Hyperspectral Bands",
            font=('Arial', 14, 'bold')
        ).pack(pady=6)
        
        # Bilgi etiketi
        self.info = ttk.Label(frm, text="---", foreground='blue')
        self.info.pack()
        
        # Görüntü alanı
        imgf = ttk.Frame(frm)
        imgf.pack(fill=tk.BOTH, expand=True, pady=6)
        
        # Sol panel - Orijinal
        lf = ttk.LabelFrame(imgf, text="Orijinal")
        lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
        self.left = ttk.Label(lf)
        self.left.pack(padx=6, pady=6)
        
        # Sağ panel - Maskelenmiş
        rf = ttk.LabelFrame(imgf, text="Yaprak (Beyaz Arka Plan)")
        rf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6)
        self.right = ttk.Label(rf)
        self.right.pack(padx=6, pady=6)
        
        # Kontrol paneli
        ctrl = ttk.Frame(frm)
        ctrl.pack(fill=tk.X, pady=6)
        
        ttk.Label(ctrl, text="Hyperspectral Band (0-223):").grid(
            row=0, column=0, sticky='w', padx=4
        )
        
        # Band slider
        self.band_var = tk.IntVar(value=self.node.current_band)
        s = ttk.Scale(
            ctrl,
            from_=0,
            to=223,
            orient=tk.HORIZONTAL,
            variable=self.band_var,
            command=self._on_slider
        )
        s.grid(row=0, column=1, sticky='ew', padx=6)
        ctrl.columnconfigure(1, weight=1)
        
        # Band değer
        self.band_label = ttk.Label(ctrl, text=str(self.node.current_band))
        self.band_label.grid(row=0, column=2, padx=6)
        
        # Band tipi
        self.band_type = ttk.Label(
            ctrl,
            text=get_band_type(self.node.current_band),
            foreground='darkgreen'
        )
        self.band_type.grid(row=0, column=3, padx=6)
        
        # Butonlar
        ttk.Button(ctrl, text="Uygula", command=self._apply_band).grid(
            row=0, column=4, padx=6
        )
        ttk.Button(ctrl, text="Sıfırla", command=self._reset_band).grid(
            row=0, column=5, padx=6
        )
        
        # Çıkış butonu
        ttk.Button(frm, text="Çıkış", command=self._quit).pack(pady=6)
    
    def _on_slider(self, _):
        """Slider değiştiğinde çağrılır"""
        b = int(self.band_var.get())
        self.band_label.config(text=str(b))
        self.band_type.config(text=get_band_type(b))
        self.node.current_band = b
        
        # Debounce
        if self.slider_after:
            self.root.after_cancel(self.slider_after)
        self.slider_after = self.root.after(250, self._apply_band)
    
    def _apply_band(self):
        """Band filtresini uygular"""
        b = int(self.band_var.get())
        
        with self.node.lock:
            src = (self.node.last_corrected.copy()
                   if self.node.last_corrected is not None
                   else (self.node.cv_image.copy()
                         if self.node.cv_image is not None
                         else None))
            mask = (self.node.last_mask.copy()
                   if self.node.last_mask is not None
                   else None)
            self.node.current_band = b
        
        if src is None:
            self.info.config(text="Görüntü yok", foreground='red')
            return
        
        # Band filtresini uygula
        filtered = apply_band_filter(src, b, self.node.verbose)
        
        with self.node.lock:
            self.node.band_filtered_image = filtered.copy()
            
            if mask is not None:
                # Maskeyi temizle ve uygula
                final_mask = clean_mask(mask)
                mask_img = apply_mask_to_image(filtered, final_mask)
                
                self.node.mask_image = mask_img
                self.node.original_image = src.copy()
                
                band_type = get_band_type(b)
                self.info.config(
                    text=f"Band {b} ({band_type}) uygulandı",
                    foreground='green'
                )
            else:
                # Maske yoksa beyaz arka plan
                white = np.ones_like(filtered) * 255
                self.node.mask_image = white.copy()
                
                band_type = get_band_type(b)
                self.info.config(
                    text=f"Band {b} ({band_type}) önbelleklendi, mask bekleniyor",
                    foreground='orange'
                )
    
    def _reset_band(self):
        """Band değerini sıfırlar"""
        self.band_var.set(112)
        self.band_type.config(text=get_band_type(112))
        self.node.current_band = 112
        self._apply_band()
    
    def _prep(self, img):
        """Görüntüyü Tkinter için hazırlar"""
        if img is None:
            return None
        
        h, w = img.shape[:2]
        maxs = 450
        
        if h > maxs or w > maxs:
            s = min(maxs / h, maxs / w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
        
        return ImageTk.PhotoImage(
            PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        )
    
    def _update_ui(self):
        """UI'yi günceller"""
        try:
            with self.node.lock:
                left = (self.node.original_image.copy()
                       if self.node.original_image is not None
                       else (self.node.cv_image.copy()
                             if self.node.cv_image is not None
                             else None))
                right = (self.node.mask_image.copy()
                        if self.node.mask_image is not None
                        else None)
            
            # Sağ görüntü yoksa beyaz arka plan
            if right is None:
                right = np.ones((480, 640, 3), dtype=np.uint8) * 255
            
            # Görüntüleri hazırla
            imgL = self._prep(left)
            imgR = self._prep(right)
            
            if imgL:
                self.left.configure(image=imgL)
                self.left.image = imgL
            
            if imgR:
                self.right.configure(image=imgR)
                self.right.image = imgR
            
            # İstatistikleri güncelle
            pt = getattr(self.node, 'processing_time', 0.0)
            fps = 1.0 / pt if pt > 0 else 0.0
            det = getattr(self.node, 'detection_count', 0)
            band_type = get_band_type(self.node.current_band)
            
            self.info.config(
                text=f"İşlem: {pt:.3f}s | FPS: {fps:.1f} | "
                     f"Band: {self.node.current_band} ({band_type}) | "
                     f"Tespit: {det}"
            )
        
        except Exception as e:
            print('UI hata:', e)
        
        self.root.after(50, self._update_ui)
    
    def _quit(self):
        """Uygulamayı kapatır"""
        try:
            self.node.stop()
        except:
            pass
        self.root.quit()


