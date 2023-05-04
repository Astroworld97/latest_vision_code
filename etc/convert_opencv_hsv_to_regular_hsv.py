hsv1_opencv = (105, 255, 187)
hsv2_opencv = (25, 255, 255)

hsv1_reg = ((hsv1_opencv[0]*360/180), (hsv1_opencv[1]*100/255), (hsv1_opencv[2]*100/255))
hsv2_reg = ((hsv2_opencv[0]*360/180), (hsv2_opencv[1]*100/255), (hsv2_opencv[2]*100/255))

print(hsv1_reg)
print(hsv2_reg)