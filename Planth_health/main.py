# Gerekli kütüphanelerin eklenmesi
import threading
from yolo_node import YOLOv8MaskNode, ROS_AVAILABLE
from gui import TkinterGUI
# Ros2 ve Python arasında köprüleme
if ROS_AVAILABLE:
    import rclpy

def main():
    """Ana fonksiyon"""
    # ROS başlat
    if ROS_AVAILABLE:
        rclpy.init()
    
    # Node oluştur
    node = YOLOv8MaskNode()
    
    # ROS spin thread
    if ROS_AVAILABLE:
        threading.Thread(
            target=rclpy.spin,
            args=(node,),
            daemon=True
        ).start()
    
    # GUI başlatma   
    gui = TkinterGUI(node)
    gui.root.mainloop()
    
    # Temizlik
    try:
        node.stop()
    except:
        pass
    
    if ROS_AVAILABLE:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()

