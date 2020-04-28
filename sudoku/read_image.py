import cv2
import numpy as np
import os
import glob
import random
import logging
from .classifier import classifier
#from classifier import classifier

#from digitRecognition.SolveSudoku.NumberExtractor import identify_number

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)  # filename='example.log',
logger = logging.getLogger(__name__)

classifier = classifier.DigitClassifier()
classifier.load(os.path.join(os.path.dirname(__file__), "classifier", "models"))

def get_best_contour(image, debug=True):
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(F"Contours found: {len(contours)}")
    best_square = None
    best_contour = None
    max_area = 400  # this is for saving temporary max value and for starting with a minimum area
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            epsilon = 0.1*cv2.arcLength(cnt, True)
            square = cv2.approxPolyDP(cnt, epsilon, True)
            if len(square) == 4:
                best_square = square.reshape((4, 2))
                max_area = area
                best_contour = cnt
    logger.debug(F"Found square: {len(best_square)}")
    if debug:
        debug_image = np.zeros((image.shape[0], image.shape[1], 3))
        cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 1)
        cv2.drawContours(debug_image, [best_contour], 0, (255, 0, 0), 3)
        cv2.drawContours(debug_image, [best_square], 0, (0, 0, 255), 1)
        cv2.imshow("Contours", debug_image)
    return best_square


def order_points(pts):
    # 1. top-left, 2. top-right, 3. bottom-right, 4. bottom-left
    rect = np.zeros((4, 2), dtype="float32")  # This has to be "float32"!
    # the top-left point will have the smallest sum, whereas the bottom-right point the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # top-right point will have the smallest difference, whereas the bottom-left the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    logger.debug(F"Order points")
    return rect


def warp_image(img, square, dimension=400):
    dst = np.array([[0, 0], [dimension - 1, 0], [dimension - 1, dimension - 1], [0, dimension - 1]], dtype="float32")
    rect = order_points(square)
    width = rect[1, 0] - rect[0, 0]
    height = rect[2, 1] - rect[0, 1]
    if 150 > width or 150 > height:
        logger.warning("Wrong Sudoku size detected")
    trans_matrix = cv2.getPerspectiveTransform(rect, dst)  # compute perspective transform matrix
    return cv2.warpPerspective(img, trans_matrix, (dimension, dimension))


def find_sudoku_in_image(path, debug=True):
    print(path)
    img = resize(cv2.imread(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logger.debug("Gaussian Blur with a filter size of 11x11")
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    logger.debug("Create Binary Image with a gaussian adaptive Threshold of 15x15")
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3)
    # get contour and the square from the contour
    fitted_square = get_best_contour(thresh, debug)
    warped = warp_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), fitted_square, dimension=450)
    if debug:
        cv2.imshow("Find Sudoku: Threshold", thresh)
        cv2.imshow("Find Sudoku: Result", warped)
    return img, warped

def read_sudoku(path, debug=True):
    image, warped = find_sudoku_in_image(path, False)
    gaussian = cv2.GaussianBlur(warped, (5, 5), 0)
    adapt_thresh = cv2.adaptiveThreshold(gaussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
    adapt_thresh = cv2.bitwise_not(adapt_thresh)
    sudoku = adapt_thresh #.copy()
    border = 4
    if debug:
        img_nums = np.zeros((450, 450, 3))# warp_image(image, fitted_square, dimension=450)  # for debugging only
    sudoku_grid = np.zeros((9, 9)).astype("uint8")
    for row in range(9): #[6]:#range(9):
        for col in range(9):#[4]:#range(9):
            image = sudoku[row*50+border:(row+1)*50-border, col*50+border:(col+1)*50-border]
            logger.debug(F"({row+1}, {col+1}) Sum: {image.sum()}")
            if image.sum() < 9000:
                continue

            contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
            num_cnt = None
            if len(contours) > 5:  # reduce number of contours
                contours = contours[0:5]
            for cnt in contours:
                moments = cv2.moments(cnt)
                cx = cy = 0
                if moments["m00"] > 0.001:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                area = moments["m00"]

                logger.debug(F"({row+1}, {col+1}) Area: {area}, Center ({cx}, {cy})")
                if 1000 > area > 30:
                    num_cnt = cnt
                    break

            if num_cnt is not None:
                num_mask = np.zeros((50 - border*2, 50 - border*2))
                cv2.fillPoly(num_mask, pts=[num_cnt], color=255)
                #cv2.imshow("mask", num_mask)
                #cv2.imshow("before ", image)
                image[num_mask == 0] = 0
                # get only the digit
                x, y, w, h = cv2.boundingRect(num_cnt)
                pad = (h-w)//2
                aspect_ratio = w / h

                logger.debug(F"Position ({x}, {y}), width: {w}, height: {h}, ratio: {aspect_ratio:.2f}")
                if aspect_ratio < 0.2 or aspect_ratio > 1.0:  # < 5 or w < 5:
                    logger.debug(F"Wrong digit size detected")
                else:
                    # cut out the digit with some padding to left and right side
                    digit = np.zeros((h, h))
                    digit[:, pad:pad+w] = image[y:y+h, x:x+w]

                    #cv2.imshow("num mask", num_mask)
                    #cv2.imshow("digit", digit)
                    #cv2.waitKey(0)
                    digit_nr = classifier.predict(digit)
                    sudoku_grid[row, col] = digit_nr
                    if debug:
                        img_nums[row*50:row*50+digit.shape[0], col*50:col*50+digit.shape[1], 1] = digit
                        cv2.putText(img_nums, str(digit_nr), (int(col*50)+5, int(row*50)+45),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

    if debug:
        cv2.imshow('adaptive thresh', adapt_thresh)
        cv2.imshow("cont mask", img_nums)
        cv2.waitKey(0)
    return sudoku_grid


def resize(img):
    height = int(800 / img.shape[0] * img.shape[1])
    return cv2.resize(img, (height, 800))


def profile_sudoku(file):
    import cProfile, pstats, time
    from io import StringIO
    pr = cProfile.Profile()
    pr.enable()
    start = time.time()
    # start of profile
    sudoku_grid = read_sudoku(file, debug=False)
    # end profile
    pr.disable()
    end = time.time()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('time')
    ps.print_stats(15)
    print(s.getvalue())
    print(F"Time for execution: {end - start}")
    print(F"File: {file}")
    print(F"{sudoku_grid}")

if __name__ == '__main__':
    image_path = os.path.join(os.path.dirname(__file__), "data")
    files = glob.glob(os.path.join(image_path, "*.jpg"))
    file = random.choice(files)
    #file = 'C:/Users/Blitzliner/Documents/git/sudokuSolver/readImage\sudoku_dataset\original\image1003.original.jpg'
    #file = 'C:/Users/Blitzliner/Documents/git/sudokuSolver/readImage\sudoku_dataset\original\image1062.original.jpg'
    #file = 'C:/Users/Blitzliner/Documents/git/sudokuSolver/readImage\sudoku_dataset\original\image1013.original.jpg'
    file = 'data/image1088.original.jpg'
    #file = 'C:/Users/Blitzliner/Documents/git/sudokuSolver/readImage\sudoku_dataset\original\image1030.original.jpg'
    #file = 'C:/Users/Blitzliner/Documents/git/sudokuSolver/readImage\sudoku_dataset\original\image1012.original.jpg'
    problems = []
    #problems.append(
    #    'C:/Users/Blitzliner/Documents/git/sudokuSolver/readImage\sudoku_dataset\original\image1087.original.jpg',
    #    'C:/Users/Blitzliner/Documents/git/sudokuSolver/readImage\sudoku_dataset\original\image1073.original.jpg' )

    sudoku_grid = read_sudoku(file, debug=False)

    #profile_sudoku(file)

    # for file in files[:3]:
    #    print(file)
    #    read(file)
