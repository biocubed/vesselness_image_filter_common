#!/usr/bin/env python
# license removed for brevity
import rospy
import cv2
import numpy as np
from vesselness_image_filter_common.msg import vesselness_params


def do_nothing(x):
    pass


def main():
    pub = rospy.Publisher("/vesselness/settings", vesselness_params, queue_size=10)
    rospy.init_node("vesselness_filter_parameter_gui", anonymous=True)
    rate = rospy.Rate(1) # 10hz
    # Initialize parameters
    params = vesselness_params()
    # Setup GUI
    window_name = "Vesselness Parameter GUI"
    image = np.zeros((1, 512))
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Hessian Side", window_name, 1, 4, do_nothing)
    cv2.createTrackbar("Hessian Variance", window_name, 0, 255, do_nothing)
    cv2.createTrackbar("Post Process Side", window_name, 1, 4, do_nothing)
    cv2.createTrackbar("Post Process Variance", window_name, 0, 255, do_nothing)
    cv2.createTrackbar("c Parameter", window_name, 0, 255, do_nothing)
    cv2.createTrackbar("beta Parameter", window_name, 0, 255, do_nothing)
    while not rospy.is_shutdown():
        cv2.imshow(window_name, image)
        k = cv2.waitKey(1) & 0xFF

        if k in [27, ord('q')]:
            rospy.signal_shutdown('Quit')

        params.hessianSide = cv2.getTrackbarPos("Hessian Side", window_name)
        params.hessianVariance = cv2.getTrackbarPos("Hessian Variance", window_name) / 10.0
        params.postProcessSide = cv2.getTrackbarPos("Post Process Side", window_name)
        params.postProcessVariance = cv2.getTrackbarPos("Post Process Variance", window_name) / 10.0
        params.cParameter = 0.001 + 0.999 / 255 * cv2.getTrackbarPos("c Parameter", window_name)
        params.betaParameter = 0.001 + 0.999 / 255 * cv2.getTrackbarPos("beta Parameter", window_name)
        rospy.loginfo(params)
        pub.publish(params)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
