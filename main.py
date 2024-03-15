# Author Furkan Ülke
# CMPE 538 - Assignment 1
# Camera Geometry Estimation
# Date: 2024-03-14

# Importing the required libraries
import matplotlib.pyplot as plt
import numpy as np

# Sample coordinates, I averaged the two samples to get a better result
"""
sample1 = [(93.00649350649348, 466.5995670995669), (393.8722943722943, 423.3095238095236),
           (183.91558441558436, 797.7683982683982), (434.9978354978356, 747.9848484848483),
           (789.9761904761905, 496.90259740259717), (980.4523809523811, 648.4177489177489),
           (753.1796536796538, 841.0584415584415), (898.2012987012987, 1038.0281385281385),
           (1118.9805194805194, 1343.2229437229437), (434.9978354978356, 1144.0887445887447),
           (99.49999999999989, 1232.8333333333333), (517.2489177489177, 1442.7900432900433),
           (114.65151515151513, 1589.9761904761906)]

sample2 = [(93.00649350649348, 464.4350649350647), (396.0367965367965, 429.80303030303),
           (188.24458874458867, 797.7683982683982), (434.9978354978356, 750.1493506493505),
           (787.8116883116883, 496.90259740259717), (980.4523809523811, 650.5822510822509),
           (757.508658008658, 843.2229437229437), (900.3658008658009, 1044.5216450216449),
           (1118.9805194805194, 1330.2359307359307), (434.9978354978356, 1139.7597402597403),
           (103.8290043290043, 1224.1753246753246), (523.7424242424241, 1442.7900432900433),
           (112.48701298701292, 1594.3051948051948)]
"""

# Corresponding real and image coordinates
image_coordinates = [(93.00649350649348, 465.5173160173158),
                     (394.9545454545454, 426.5562770562768),
                     (186.08008658008652, 797.7683982683982),
                     (434.9978354978356, 749.0670995670994),
                     (788.8939393939394, 496.90259740259717),
                     (980.4523809523811, 649.4999999999999),
                     (755.3441558441559, 842.1406926406926),
                     (899.2835497835498, 1041.2748917748918),
                     (1118.9805194805194, 1336.7294372294373),
                     (434.9978354978356, 1141.9242424242425),
                     (101.6645021645021, 1228.504329004329),
                     (520.4956709956709, 1442.7900432900433),
                     (113.56926406926402, 1592.1406926406926)
                     ]

real_coordinates = [(29.5, 0, 40),
                    (14.5, 0, 40),
                    (29.5, 0, 20),
                    (14.5, 0, 20),
                    (0, 15, 40),
                    (0, 30, 40),
                    (0, 15, 20),
                    (0, 30, 20),
                    (0, 45, 20),
                    (19.5, 12.5, 0),
                    (39.5, 12.5, 0),
                    (19.5, 32.5, 0),
                    (39.5, 32.5, 0)
                    ]

# Coordinates of the vertices of the cube that we want to place in the image
imaginary_coordinates = np.array([
    [0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0],  # Bottom face
    [0, 0, 20], [20, 0, 20], [20, 20, 20], [0, 20, 20]  # Top face
])

# Define the edges of the cube as indices of the vertices that form each edge
edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face edges
    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face edges
    [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical edges connecting bottom and top faces
]


# 3D to 2D projection
def project_3d_to_2d(m, point3d):
    homogeneous_3d = np.append(point3d, 1)
    homogeneous_2d = np.dot(m, homogeneous_3d)
    cartesian_2d = homogeneous_2d / homogeneous_2d[-1]
    x_point, y_point, _ = cartesian_2d
    return x_point, y_point


# Calculate the projection row for a single point correspondence
def calculate_projection_row(image_coordinate, real_coordinate):
    result = [(real_coordinate[0], real_coordinate[1], real_coordinate[2], 1, 0, 0, 0, 0,
               -image_coordinate[0] * real_coordinate[0], -image_coordinate[0] * real_coordinate[1],
               -image_coordinate[0] * real_coordinate[2], -image_coordinate[0]),
              (0, 0, 0, 0, real_coordinate[0], real_coordinate[1], real_coordinate[2], 1,
               -image_coordinate[1] * real_coordinate[0], -image_coordinate[1] * real_coordinate[1],
               -image_coordinate[1] * real_coordinate[2], -image_coordinate[1])]

    return result


# Function to handle mouse click events. It appends the clicked coordinates to the image_coordinates list
# Used only for the first time to get the image coordinates
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        image_coordinates.append((event.xdata, event.ydata))
        print(f'Clicked at ({event.xdata: .2f}, {event.ydata: .2f})')


# Function to calculate the piecewise average of two arrays
def piecewise_average(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length")

    result = []
    for (x1, y1), (x2, y2) in zip(array1, array2):
        avg_x = (x1 + x2) / 2
        avg_y = (y1 + y2) / 2
        result.append((avg_x, avg_y))

    return result


if __name__ == "__main__":
    # This part is used to get the image coordinates of selected points
    # Only used once to get the image coordinates
    # Uncomment this part and run the code to get your own image coordinates
    """image = plt.imread('tiles.jpeg')

    plt.imshow(image)
    plt.title('Click on the image to get coordinates.')
    plt.axis('on')

    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    plt.show()
    image_coordinates = piecewise_average(sample1, sample2)
    """

    if len(image_coordinates) != len(real_coordinates):
        raise ValueError("Arrays must have the same length")

    A = []
    # Calculate the projection matrix row by row for each point correspondence
    for image_coordinate_sample, real_coordinate_sample in zip(image_coordinates, real_coordinates):
        A.extend(calculate_projection_row(image_coordinate_sample, real_coordinate_sample))

    A = np.array(A)

    # Perform SVD to get the camera matrix
    U, S, V = np.linalg.svd(A)

    # The camera matrix is the last row of V
    camera_matrix = V[-1, :].reshape(3, 4)

    print("Camera Matrix:"
          "\n", camera_matrix)

    projected_imaginary = []
    # Project the 3D coordinates of the cube to 2D image coordinates
    for point in imaginary_coordinates:
        x, y = project_3d_to_2d(camera_matrix, point)
        projected_imaginary.append((x, y))

    num_points_to_interpolate = 100
    interpolated_points = []
    # Interpolate the lines between the projected 2D points to get a better visualization
    for edge in edges:
        start = projected_imaginary[edge[0]]
        end = projected_imaginary[edge[1]]
        line_points = np.linspace(start, end, num_points_to_interpolate + 1)
        interpolated_points.extend(line_points[:-1])

    # Convert the list of interpolated points to a numpy array
    interpolated_points = np.array(interpolated_points)
    image = plt.imread('tiles.jpeg')
    plt.imshow(image)

    # Plot the projected 2D points and the interpolated lines
    for edge in edges:
        start_point = projected_imaginary[edge[0]]
        end_point = projected_imaginary[edge[1]]
        plt.plot([start_point[0], end_point[0]],
                 [start_point[1], end_point[1]], color='r')

    # Plot the interpolated points
    plt.scatter(interpolated_points[:, 0], interpolated_points[:, 1], color='b', s=5)
    plt.title('Interpolated Points on Original Image')

    # Show the plot
    plt.show()

