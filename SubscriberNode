#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def calculate_distance(contour_area, focal_length=700, real_width=0.2):
    """
    Рассчитывает расстояние до объекта, используя известный размер и площадь контура.

    :param contour_area: площадь найденного контура
    :param focal_length: фокусное расстояние камеры в пикселях (определяется экспериментально)
    :param real_width: реальная ширина объекта в метрах
    :return: расстояние до объекта в метрах
    """
    if contour_area <= 0:
        return float('inf')  # Если контуров нет, возвращаем бесконечное расстояние
    
    return (focal_length * real_width) / np.sqrt(contour_area)

def callback(data):
    br = CvBridge()
    rospy.loginfo("Receiving video frame")
    
    frame = br.imgmsg_to_cv2(data)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)

        distance = calculate_distance(contour_area)

        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        cv2.putText(frame, f"Distance: {distance:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)

    cv2.imshow("Processed Camera Feed", frame)
    cv2.waitKey(1)

def receive_message():
    rospy.init_node('video_sub_py', anonymous=True)
    rospy.Subscriber('video_frames', Image, callback)
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    receive_message()
