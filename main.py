import numpy as np            # For numPy arrays
import cv2                    # For webcam and other stuff
import math                   # For cosine maths
import imutils as imutils     # For resizing the frame
import winsound               # For beep sound. Not needed now
from threading import Thread  # For creating threads


# Change color of pixels in all octants
# Called by circleBres
def drawCircle(xc, yc, x, y, img):
    # putpixel(xc+x, yc+y, RED);
    img[yc + y, xc + x, 0] = 0
    img[yc + y, xc + x, 1] = 0
    img[yc + y, xc + x, 2] = 255
    # putpixel(xc-x, yc+y, RED);
    img[yc + y, xc - x, 0] = 0
    img[yc + y, xc - x, 1] = 0
    img[yc + y, xc - x, 2] = 255
    # putpixel(xc+x, yc-y, RED);
    img[yc - y, xc + x, 0] = 0
    img[yc - y, xc + x, 1] = 0
    img[yc - y, xc + x, 2] = 255
    # putpixel(xc-x, yc-y, RED);
    img[yc - y, xc - x, 0] = 0
    img[yc - y, xc - x, 1] = 0
    img[yc - y, xc - x, 2] = 255
    # putpixel(xc+y, yc+x, RED);
    img[yc + x, xc + y, 0] = 0
    img[yc + x, xc + y, 1] = 0
    img[yc + x, xc + y, 2] = 255
    # putpixel(xc-y, yc+x, RED);
    img[yc + x, xc - y, 0] = 0
    img[yc + x, xc - y, 1] = 0
    img[yc + x, xc - y, 2] = 255
    # putpixel(xc+y, yc-x, RED);
    img[yc - x, xc + y, 0] = 0
    img[yc - x, xc + y, 1] = 0
    img[yc - x, xc + y, 2] = 255
    # putpixel(xc-y, yc-x, RED);
    img[yc - x, xc - y, 0] = 0
    img[yc - x, xc - y, 1] = 0
    img[yc - x, xc - y, 2] = 255


# Called when 5 fingers are detected
# Make big white rectangles over img
def eraseCircles(x, y, img):
    img[x:x + 200, y:y + 200, 0] = 255
    img[x:x + 200, y:y + 200, 1] = 255
    img[x:x + 200, y:y + 200, 2] = 255


# Implement Bresenham's circle algorithm
# Called when 1, 2, 3, or 4 fingers are detected
def circleBres(xc, yc, rad, img):
    img[yc - 3: yc + 3, xc - 3: xc + 3, 0] = 0
    img[yc - 3: yc + 3, xc - 3: xc + 3, 1] = 0
    img[yc - 3: yc + 3, xc - 3: xc + 3, 2] = 255

    for r in range(20, rad, 20):
        x = 0
        y = r
        d = 3 - 2 * r
        drawCircle(xc, yc, x, y, img)
        while y >= x:
            # for each pixel we will
            # draw all eight pixels
            x = x + 1

            # check for decision parameter
            # and correspondingly
            # update d, x, y
            if d > 0:
                y = y - 1
                d = d + 4 * (x - y) + 10
            else:
                d = d + 4 * x + 6
            drawCircle(xc, yc, x, y, img)

    # plt.imshow(img, interpolation='nearest')
    # plt.show()


# Make beep sound
# Just for fun
# In case you wanna blow horn, call this function
def playBeep():
    frequency = 470
    duration = 300
    winsound.Beep(frequency, duration)


# Prints message on webcam stream
def hide(msg):
    cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)


# Driver code starts here
# Open Camera
webcam = cv2.VideoCapture(0)
# count_hide = 0

# Read the required image from PC and store it in img
img = cv2.imread('C:/Users/Saniya/Desktop/Coding Stuff/whiteImg.png')

# Change BGR to RGB
# Not required here
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Iterate while webcam is opened
while webcam.isOpened():

    # Read the webcam and store each frame in the variable frame
    ret, frame = webcam.read()

    # Resize frame to 800, 800
    frame = imutils.resize(frame, width=800, height=800)

    # Get hand data from the rectangle sub window
    # This statement makes green rectangle around your hand
    cv2.rectangle(frame, (0, 0), (500, 450), (0, 255, 0), 4)
    # This statement crops required frame from the webcam, i.e., the green
    # rectangle formed in above statement and store it in crop_image
    crop_image = frame[0: 500, 0: 450]

    # Apply Gaussian blur
    # Now what is Gaussian Blur? Well it basically blurs the nonsense
    # things in the back, in order to just capture your beautiful hand
    blur = cv2.GaussianBlur(crop_image, (5, 5), 0)

    # Change color-space from BGR -> HSV
    # Now what is HSV? Nothing, it just means we ware converting the image
    # in numerical form
    # It also saturates the image. Saturtion is like acrylic painting, with
    # all the colors merged into each other. So basically, making your
    # hand more beautiful.
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    # This will mask the hsv image. Masking is converting the image into binary form
    # the colors, we need that is skin color, will be white and the rest will be black
    # in mask2
    # This np arrays contain the BGR codes for darkest and brightest skin color
    # Do no search them, its very racist
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Kernel for morphological transformation
    # Now this is big brain stuff
    kernel = np.ones((5, 5))
    # Apply morphological transformations to filter out the background noise
    # In simple language, we just wanna focus on the hand, so we increase the size
    # of hand using dilation and remove the bg noises, i.e., unwanted stuff from the
    # background by erosion.
    # Remember erosion also decreases the size of our hand, so we use dilation
    # kernel is just something that is required, I don't know why
    # Remember this stuff can only be done on binary images
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    # Now, we do Gaussian Blur again. Blur is done to get sharp edges
    filtered = cv2.GaussianBlur(erosion, (5, 5), 0)

    # Here we set a threashold and a maximum value
    # If the pixel of the image is less than threshold, then set it to 0
    # Else set it to max value, which here is 200
    thresh = cv2.threshold(filtered, 127, 200, 0)[1]

    # Show threshold image
    # This image will show your hand white, and other stuff as black
    # This will be updated every frame, so you will actually see threshold webcam stream
    cv2.imshow("Threshold", thresh)

    # Find contours
    # Now, if you do makeup, you know that a contour is basically used to highlight
    # your cheekbones and other stuff. Similarly here, contour makes an outline around
    # the image that has the same intensity, i.e., your hand
    # So, here we are just extracting that from thresh
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Find contour with maximum area
        # This is self explanatory
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
        # x, y is top lft coordinate and w, h is bottom right coordinate
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Find convex hull
        # This might need a little understanding
        # Convex are objects with no interior angles greater than 180
        # Hull is shape of object. So here, conver hull is a convex outline of your hand
        hull = cv2.convexHull(contour)

        # This stuff is not really needed for final project but is important, so do not delete it
        # Draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Find convexity defects
        # What are convexity defects?
        # If we take example of your hand, then convexity defects are those cavities between your fingers
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points
        # (the finger tips) for all defects
        count_defects = 0

        # Here we get the number of defects/number of fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # These are predefined formulas. I don't know where they came from
            # Some genius person made them, we are just reusing
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            # We get the angle between fingers
            # If the angle > 90, then those are not fingers
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # if angle > 90 draw a circle at the far point
            # angle > 90, means it is not a finger cavity, like the angle between your pinky finger
            # and hand
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 0], 2)

        # If you got 1, 2, 3 or 4 fingers or no fingers, then execute this peice of code
        # You can draw circles on image
        if count_defects <= 3:
            # Creating threads here, so that webcam stream doesn't lag
            # Prints Circles on webcam stream
            Thread(target=hide("Circles")).start()

            # Some stuff I didn't wanna delete
            # img = cv2.imread('C:/Users/Saniya/Desktop/Coding Stuff/birthday1.png')
            # count_hide = count_hide + 1
            # if count_hide == 1:
            #     Thread(target=playBeep).start()

            # If the tip of your finger is in below range, then you can draw circles
            if 100 < x < 300 and 100 < y < 300:
                Thread(target=circleBres, args=(x, y, 50, img)).start()
            # print(x, ", ", y)

        # If your hand is open, you can erase those circles you just made
        elif count_defects >= 4:
            Thread(target=hide("Erase")).start()
            if 10 < x < 570 and 10 < y < 570:
                Thread(target=eraseCircles, args=(x, y, img)).start()

        # If this above exhausting code can not detect anything, then pass
        else:
            pass
    # If some error happens in this above code, then pass not crash
    except:
        pass

    # Show required images

    # Shows your hand in webcam, frame by frame
    # In easy language, shows the webcam stream
    cv2.imshow("Gesture", frame)

    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)

    # Show img with you drawing and erasing those red concentric circles
    cv2.imshow("Image", img)

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Clear all windows
webcam.release()
cv2.destroyAllWindows()
