import matplotlib.pyplot as plt
import numpy as np
import imageio as i
import math
import cv2
import glob

def GradientOperator(img, op):
    ans = np.zeros_like(img,dtype=float) #gx and gy output
    image_padded = np.zeros((img.shape[0]+2, img.shape[1]+2)) #Add zero padding to the input image
    image_padded[1:-1, 1:-1] = img
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            #element-wise multiplication of respective horizontal and vertical operator and the image
            if (x<1 or x>=img.shape[0]-1) or (y<1 or y>=img.shape[1]-1):
                ans[x,y]=0 
                #pixel values of first 4 rows 4 columns and last 4 rows 4 columns will be undefined
            else:
                ans[x,y]=(np.sum(op*image_padded[x:x+3,y:y+3]))/3 
                #normalised by dividing by 3
    return ans

def histogram(angles, magnitudes):
    # [0, 20, 40, 60, 80, 100, 120, 140, 160]
    h = np.zeros(9, dtype=np.float32)

    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            if int(angles[i,j])<0:
                index_1 = 8
                index_2 = 0

                proportion = (index_2 * 20 - angles[i, j]) / 20

                value_1 = proportion * magnitudes[i, j]
                value_2 = (1 - proportion) * magnitudes[i, j]
                h[index_1] += value_1
                h[index_2] += value_2
                
            elif int(angles[i,j])>=160:
                index_1 = 0
                index_2 = 8

                proportion = (angles[i, j]-index_2 * 20) / 20

                value_1 = proportion * magnitudes[i, j]
                value_2 = (1 - proportion) * magnitudes[i, j]
                h[index_1] += value_1
                h[index_2] += value_2
            else:
                index_1 = int(angles[i, j] // 20)
                index_2 = int(angles[i, j] // 20 + 1)

                proportion = (index_2 * 20 - angles[i, j]) / 20

                value_1 = proportion * magnitudes[i, j]
                value_2 = (1 - proportion) * magnitudes[i, j]
                h[index_1] += value_1
                h[index_2] += value_2
    return h

def cells(trya,g):
    #creating cells of size 8x8
    cells = []
    cell_size=8

    for i in range(0, np.shape(trya)[0],cell_size):
        row = []
        for j in range(0, np.shape(trya)[1],cell_size):
            row.append(np.array(
                    histogram(trya[i:i + cell_size, j:j + cell_size], g[i:i + cell_size, j:j + cell_size]),
                    dtype=np.float32))
            cells.append(row)
            return cells
        
def hog_descriptor(cells):
    #creating final hog vector
    hog_vector = []
    for i in range(0,np.shape(cells)[0]-1):
        for j in range(0,np.shape(cells)[1]-1):
            block_vector = []
            block_vector.extend(cells[i][j])
            block_vector.extend(cells[i][j + 1])
            block_vector.extend(cells[i + 1][j])
            block_vector.extend(cells[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)
    hog = [item for sublist in hog_vector for item in sublist]
    return hog

def p1(image):
    #lets say this is the main function for the first part of the project i.e calculating HoG
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    img = 0.299 * r + 0.587 * g + 0.114 * b
    px = np.array([[-1, 0, 1], 
                   [-1, 0, 1], 
                   [-1, 0, 1]]) #horizontal operator
    py = np.array([[1, 1, 1], 
                   [0, 0, 0], 
                   [-1, -1, -1]]) #vertical operator
    gx = GradientOperator(img, px) #horizontal gradient
    gx1=abs(gx) #we take absolute values for display purpose
    gy = GradientOperator(img, py) #vertical gradient
    gy1=abs(gy)
    g = (np.sqrt((gx1 * gx1) + (gy1 * gy1))/math.sqrt(2)) #normalise the magnitude by root(2)
    prewitt = (np.arctan2(gy, gx) * (180/np.pi)) #calculate the edge angles

    trya = np.copy(prewitt)
    for i in range(trya.shape[0]):
        for j in range(trya.shape[1]):
            if trya[i,j]<-10:
                trya[i,j]=360+trya[i,j]
            if trya[i,j]>=170 and trya[i,j]<350:
                trya[i,j]=trya[i,j]-180
    
    cells_c = cells(trya,g) 
    hog_val = hog_descriptor(cells_c)
    return hog_val


