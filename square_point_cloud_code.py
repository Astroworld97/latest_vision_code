# Note to self: remember images are indexed as rows and columns. Use r and c to iterate, rather than x and y.
### y = rows = r###
### x = columns = c###

# # Create point cloud
# point_cloud = []

# # reconstruct original image as plot
# # Create an empty plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# keys = list(colorDict.keys())
# print(len(keys))
# # Loop through each pixel and set its color using the given HSV values
# for point in keys:
#     # Get the HSV values for the current pixel
#     hsv = colorDict[tuple(point)]
#     h = hsv[0]
#     s = hsv[1]
#     v = hsv[2]

#     # Convert the HSV values to RGB values
#     r, g, b = colorsys.hsv_to_rgb(h, s, v)
#     color = np.array([r, g, b])
#     color = color.reshape(1, -1)

#     # Set the color of the pixel (aka point) in the image using the RGB values
#     col = point[0]
#     row = point[1]
#     z = point[2]
#     ax.scatter(col, height-row, z, c=color, marker='o')
# # Set the axis limits and labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# save the plot as a pdf
# plt.savefig('point_cloud' + str(count) + '.pdf')

# # Show the plot
# plt.show()
