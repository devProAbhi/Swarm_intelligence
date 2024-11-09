import numpy as np
import random
import time
from PIL import Image
import matplotlib.pyplot as plt
import os

# Ensure the images directory exists
os.makedirs("images", exist_ok=True)
numbots = 3
img = None
botPose = []
obstaclePose = []
greenZone = []
originalGreenZone = []


def set_new_map():
    global botPose, obstaclePose, greenZone, originalGreenZone, img, numbots
    
    # Initialize/reset variables
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    botPose = []
    obstaclePose = []
    greenZone = []
    originalGreenZone = []
    score = 0
    size1 = 200
    size2 = 40
    size = size2 // 2
    arr = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    random.seed(time.time())
    
    xTop = 0
    while xTop < size1:
        yTop = 0
        while yTop < size1:
            num = random.randint(0, 3)
            newX = xTop + arr[num][0] * (size2 // 2)
            newY = yTop + arr[num][1] * (size2 // 2)
            if (newX == 0 and newY == 0) or (newX == size1 - (size2 // 2) and newY == size1 - (size2 // 2)):
                pass
            else: 
                srand = np.random.random()
                if srand < 0.3:
                    # Green zone
                    img[newX:newX + (size2 // 2), newY:newY + (size2 // 2), :] = [0, 255, 0]
                    greenZone.append([[newX, newY], [newX, newY + size - 1], [newX + size - 1, newY + size - 1], [newX + size - 1, newY]])
                else:
                    # Obstacle
                    img[newX:newX + (size2 // 2), newY:newY + (size2 // 2), :] = [0, 0, 0]
                    obstaclePose.append([[newX, newY], [newX, newY + size - 1], [newX + size - 1, newY + size - 1], [newX + size - 1, newY]])
            yTop += size2
        xTop += size2

    # Place bots in random positions on the map
    for i in range(numbots):
        x, y = random.randint(0, 199), random.randint(0, 199)
        while [x, y] in botPose or not np.all(img[x, y] == [255, 255, 255]):
            x, y = random.randint(0, 199), random.randint(0, 199)
        botPose.append([x, y])


    # Adding a green goal zone at the bottom-right corner
    img[img.shape[0] - 3:img.shape[0], img.shape[1] - 3:img.shape[1]] = [0, 255, 0]
    greenZone.append([[img.shape[0] - 3, img.shape[1] - 3], [img.shape[0] - 3, img.shape[1] - 1], [img.shape[0] - 1, img.shape[1] - 1], [img.shape[0] - 1, img.shape[1] - 3]])
    
    # Shuffle zones to randomize placements further
    random.shuffle(greenZone)
    random.shuffle(obstaclePose)
    originalGreenZone = greenZone[:]
    
    # Display the map
    plt.imshow(img)
    plt.title("Generated Grid Map")
    #plt.show()

    return img

def getMap():
    global img, numbots
    curr_img = np.copy(img)
    
    # Highlight each bot's position
    for botId in range(numbots):
        x, y = botPose[botId]
        curr_img[botPose[botId][0]-3:min(botPose[botId][0]+3,200), botPose[botId][1]-3:min(botPose[botId][1]+3,200)] = np.array([0, 0, 255])
    im=Image.fromarray(curr_img)
    im.save("images/curr_map.png")
    print("Map saved to images/curr_map.png")
    return curr_img

def get_obstacles_list():
    global obstaclePose
    return obstaclePose
def get_greenZone_list():
    global greenZone
    return greenZone
def get_original_greenZone_list():
    global originalGreenZone
    return originalGreenZone
def get_botPose_list():
    global botPose
    return botPose





