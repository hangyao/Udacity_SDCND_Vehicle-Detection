# This file includes all helper functions for
# CarND-Advanced-Lane-Lines project
import numpy as np
import cv2
import matplotlib.pyplot as plt
import Line

def undistort(img, mtx, dist):
    '''
    This function correct image based on camera calibration.
    Variables:
        img: input image
        mtx: camera matrix
        dist: distortion coefficients
    Return:
        dst: corrected image
    '''
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def warp(img):
    '''
    This funtion performs a perspective transformation to an image.
    Variable:
        img: input image
    Return:
        warped: perspective transformed image
        Minv: inverse of perspective transform matrix
    '''
    img_size = (img.shape[1], img.shape[0])
    # Four source coordinates
    src = np.float32([[545, 460],
                    [735, 460],
                    [1280, 700],
                    [0, 700]])
    # Four desired coordinates
    dst = np.float32([[0, 0],
                     [1280, 0],
                     [1280, 720],
                     [0, 720]])
    # Compute the perspective transform, M, given source and destination points
    M = cv2.getPerspectiveTransform(src, dst)
    # Compute the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp an image using the perspective transform, M
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, Minv

def color_thred(img):
    '''
    This function thresholds image based on colorspace channels.
    Variable:
        img: input image
    Return:
        binary_hls_l: binary threshold per L-channel of HLS
        binary_lab_b: binary threshold per b-channel of Hab
        combined_binary: combined binary thresholds
    '''
    # Threshold the L-channel of HLS
    hls_l = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,1]
    binary_hls_l = np.zeros_like(hls_l)
    binary_hls_l[(hls_l > 210) & (hls_l <= 255)] = 1

    # Thresholds the B-channel of LAB
    lab_b = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:,:,2]
    binary_lab_b = np.zeros_like(lab_b)
    binary_lab_b[(lab_b > 145) & (lab_b <= 255)] = 1

    # Combine all color space binary thresholds
    combined_binary = np.zeros_like(hls_l)
    combined_binary[(binary_hls_l == 1) | (binary_lab_b == 1)] = 1
    return binary_hls_l, binary_lab_b, combined_binary

def grad_thred(img):
    '''
    This function thresholds image based on Sobel gradient on x direction.
    Variable:
        img: input image
    Return:
        binary_sobelx: binary threshold per Sobel
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Threshold Sobel x gradient
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5))
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    binary_sobelx = np.zeros_like(scaled_sobelx)
    binary_sobelx[(scaled_sobelx >= 20) & (scaled_sobelx <= 255)] = 1
    return binary_sobelx

def combine_threds(img):
    '''
    This function thresholds image based on combined thresholds
    of colorspace and gradient.
    Variable:
        img: input image
    Return:
        combined_binary: combined binary threshold
    '''
    _, _, binary_color = color_thred(img)
    binary_grad = grad_thred(img)
    # Combine all color space binary thresholds
    combined_binary = np.zeros_like(binary_grad)
    combined_binary[(binary_color == 1) | (binary_grad == 1)] = 1
    return combined_binary

def locate_lanes(binary_warped):
    '''
    This function locates the lane lines.
    Variable:
        binary_warped: binary warped image
    Return:
        leftx: x coordinates of pixels on left lane
        lefty: y coordinates of pixels on left lane
        rightx: x coordinates of pixels on right lane
        righty: y coordinates of pixels on right lane
    '''
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        #              (win_xleft_high,win_y_high),(0,255,0), 2)
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),
        #              (win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty

def fit_poly(leftx, lefty, rightx, righty):
    '''
    This function fits the polynomial of lane lines.
    Variables:
        leftx: x coordinates of pixels on left lane
        lefty: y coordinates of pixels on left lane
        rightx: x coordinates of pixels on right lane
        righty: y coordinates of pixels on right lane
    Return:
        left_fit: polynomial parameters of left lane
        right_fit: polynomial parameters of right lane
    '''
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

def locate_lanes_skip_window(binary_warped, left_fit, right_fit):
    '''
    This function locates the lane lines without sliding windows.
    Variable:
        binary_warped: binary warped image
        left_fit: polynomial parameters of left lane
        right_fit: polynomial parameters of right lane
    Return:
        leftx: x coordinates of pixels on left lane
        lefty: y coordinates of pixels on left lane
        rightx: x coordinates of pixels on right lane
        righty: y coordinates of pixels on right lane
    '''
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox >
                       (left_fit[0]*(nonzeroy**2) +
                        left_fit[1]*nonzeroy +
                        left_fit[2] - margin)) &
                      (nonzerox <
                       (left_fit[0]*(nonzeroy**2) +
                        left_fit[1]*nonzeroy +
                        left_fit[2] + margin)))
    right_lane_inds = ((nonzerox >
                        (right_fit[0]*(nonzeroy**2) +
                         right_fit[1]*nonzeroy +
                         right_fit[2] - margin)) &
                       (nonzerox <
                        (right_fit[0]*(nonzeroy**2) +
                         right_fit[1]*nonzeroy +
                         right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty

def fit_poly_plot(
        binary_warped, left_fit, right_fit, leftx, lefty, rightx, righty):
    '''
    This function plots the fitted polynomials of lane lines.
    Variable:
        binary_warped: binary warped image
        left_fit: polynomial parameters of left lane
        right_fit: polynomial parameters of right lane
        leftx: x coordinates of pixels on left lane
        lefty: y coordinates of pixels on left lane
        rightx: x coordinates of pixels on right lane
        righty: y coordinates of pixels on right lane
    Return:
        result: result image
        ploty: fitted y coordinates
        left_fitx: fitted x coordinates of left lane
        right_fitx: fitted x coordinates of right lane
    '''
    margin = 100
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array(
        [np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return result, ploty, left_fitx, right_fitx

def fit_poly_m(leftx, lefty, rightx, righty):
    '''
    This function fits the polynomial of lane lines in meters.
    Variables:
        leftx: x coordinates of pixels on left lane
        lefty: y coordinates of pixels on left lane
        rightx: x coordinates of pixels on right lane
        righty: y coordinates of pixels on right lane
    Return:
        left_fit: polynomial parameters of left lane in meters
        right_fit: polynomial parameters of right lane in meters
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/800 # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    return left_fit, right_fit

def get_curv(binary_warped, left_fit, right_fit):
    '''
    This function gets the radius of curvature.
    Variable:
        binary_warped: binary warped image
        left_fit: polynomial parameters of left lane
        right_fit: polynomial parameters of right lane
    Return:
        left_curverad: radius of curvature for left lane
        right_curverad: radius of curvature for right lane
    '''
    y_eval = np.float32(binary_warped.shape[0] - 1)
    left_curverad = ((1 + (2*left_fit[0]*y_eval +
                    left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval +
                     right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return left_curverad, right_curverad

def get_curv_m(binary_warped, left_fit, right_fit):
    '''
    This function gets the radius of curvature in meters.
    Variable:
        binary_warped: binary warped image
        left_fit: polynomial parameters of left lane in meters
        right_fit: polynomial parameters of right lane in meters
    Return:
        left_curverad: radius of curvature for left lane in meters
        right_curverad: radius of curvature for right lane in meters
    '''
    y_eval = np.float32(binary_warped.shape[0] - 1)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/800 # meters per pixel in x dimension

    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix +
                           left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix +
                            right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return left_curverad, right_curverad

def dist2center_m(binary_warped, left_fit, right_fit):
    '''
    This function gets the distance to road center in meters.
    Variable:
        binary_warped: binary warped image
        left_fit: polynomial parameters of left lane
        right_fit: polynomial parameters of right lane
    Return:
        dist: distance to road center in meters
    '''
    y_eval = np.float32(binary_warped.shape[0] - 1)
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/800 # meters per pixel in x dimension
    leftx = left_fit[0]*(y_eval)**2 + left_fit[1]*y_eval+ left_fit[2]
    rightx = right_fit[0]*(y_eval)**2 + right_fit[1]*y_eval + right_fit[2]
    center_px = binary_warped.shape[1] / 2.
    left_dist = (center_px - leftx) * xm_per_pix
    right_dist = (rightx - center_px) * xm_per_pix
    return left_dist, right_dist

def project_lines(undist, warped, Minv, ploty, left_fitx, right_fitx):
    '''
    This function projects fitted lane lines onto original image.
    Variable:
        undist: original image
        warped: warped image
        Minv: inverse of perspective transform matrix
        ploty: fitted y coordinates
        left_fitx: fitted x coordinates of left lane
        right_fitx: fitted x coordinates of right lane
    Return:
        result: result image
    '''
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


def validate_lane(left_curv, right_curv, left2c, right2c, left_fit, right_fit):
    '''
    This function validate the lane lines through checking curvature,
    horizontal distance, and parallel.
    Variable:
        left_curve: radius of curvature of left lane
        right_curve: radius of curvature of right lane
        left2c: left lane to center distance
        right2c: right lane to center distance
        left_fit: quadratic polynomial parameters of left lane
        right_fit: quadratic polynomial parameters of right lane
    Return:
        flag: True: lane lines are valid; False: lane lines are invalid.
    '''
    # check curvature
    if left_curv >= 200 and right_curv >= 200:
        flag_curv = True
    else:
        flag_curv = False
    # check horizontal distance
    dist = left2c + right2c
    if dist >= 3. and dist <= 6.:
        flag_dist = True
    else:
        flag_dist = False
    # check parallel
    left_slope = left_fit[0]
    right_slope = right_fit[0]
    if np.absolute(right_slope - left_slope) <= 9e-4:
        flag_paral = True
    else:
        flag_paral = False
    return (flag_curv and flag_dist and flag_paral)

def process_image(img, mtx, dist, left, right):
    '''
    This function defines the image process pipeline.
    Variable:
        img: input image
        mtx: camera matrix
        dist: distortion coefficients
        left: left lane instance
        right: right lane instance
    Return:
        result: output image
    '''
    img0 = undistort(img, mtx, dist)
    img, Minv = warp(img0)
    img = combine_threds(img)

    if left.detected and right.detected:
        left_x, left_y, right_x, right_y = locate_lanes_skip_window(
            img, left.current_fit, right.current_fit)
        left_fitparam, right_fitparam = fit_poly(left_x, left_y, right_x, right_y)
        left_fit_m, right_fit_m = fit_poly_m(left_x, left_y, right_x, right_y)
        _, ploty, left_xfit, right_xfit = fit_poly_plot(
            img, left_fitparam, right_fitparam, left_x, left_y, right_x, right_y)
        left_curv, right_curv = get_curv_m(img, left_fit_m, right_fit_m)
        left_dist, right_dist = dist2center_m(img, left_fitparam, right_fitparam)

        if validate_lane(
            left_curv, right_curv, left_dist, right_dist, left_fitparam, right_fitparam):
            left.update_line(left_x, left_y, left_fitparam, left_xfit, left_curv, left_dist)
            right.update_line(right_x, right_y, right_fitparam, right_xfit, right_curv, right_dist)
        else:
            left.detected = False
            right.detected = False

    else:
        left_x, left_y, right_x, right_y = locate_lanes(img)
        left_fitparam, right_fitparam = fit_poly(left_x, left_y, right_x, right_y)
        left_fit_m, right_fit_m = fit_poly_m(left_x, left_y, right_x, right_y)
        _, ploty, left_xfit, right_xfit = fit_poly_plot(
            img, left_fitparam, right_fitparam, left_x, left_y, right_x, right_y)
        left_curv, right_curv = get_curv_m(img, left_fit_m, right_fit_m)
        left_dist, right_dist = dist2center_m(img, left_fitparam, right_fitparam)

        if validate_lane(left_curv, right_curv, left_dist, right_dist, left_fitparam, right_fitparam):
            left.update_line(left_x, left_y, left_fitparam, left_xfit, left_curv, left_dist)
            right.update_line(right_x, right_y, right_fitparam, right_xfit, right_curv, right_dist)
        else:
            left.detected = False
            right.detected = False

    result = project_lines(img0, img, Minv, ploty, left.bestx, right.bestx)
    curverad = (left.radius_of_curvature + right.radius_of_curvature) / 2
    if left.line_base_pos < right.line_base_pos:
        str_side = 'left'
    else:
        str_side = 'right'
    dist2center = np.absolute((right.line_base_pos - left.line_base_pos) / 2.)

    cv2.putText(result,
                'Radius of Curvature = {:.0f}m'.format(curverad),
                (100, 50),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale = 4,
                color=(255,255,255),
                thickness=2,)
    cv2.putText(result,
                'Vehicle is {:.2f}m {:s} of center'.format(dist2center, str_side),
                (100, 100),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale = 4,
                color=(255,255,255),
                thickness=2,)
    return result
