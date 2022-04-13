import pandas as pd
import numpy as np
import math
import pygame
import random
from pygame.locals import *
import time
from scipy.stats import rankdata
from pandas import Series
import pandas as pd
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import sgd
import json

# --------------Is it done?------------

def test_done(A) :
    Done = True
    for i in range(1, 5) :
        for j in range(1,5) :
            if (A[i, j+1] == (0 or A[i, j])) or (A[i-1, j] == (0 or A[i, j])) or (A[i, j-1] == (0 or A[i, j])) or (A[i+1, j] == (0 or A[i, j])) :
                Done = False
                break
    return Done

# ----------Random generation of 2's---------------

def two_generator(A) :
    two = True
    rand_four = random.randint(1, 10)
    while two :
        new_x = random.randint(1, 4)
        new_y = random.randint(1, 4)
        if A[new_x, new_y] == 0 :
            if rand_four == 1 :
                A[new_x, new_y] = 4
                two = False
            else :
                A[new_x, new_y] = 2
                two = False

# --------------Creation of the input Matrix (one hot encoded)----------------------

def input_matrix_creation(A, Values) :
    input = 176
    square_num = 0
    for i in range(1, 5) :
        for j in range(1, 5) :
            for k in range(1, 12) :
                if A[i, j] == 2**k :
                    Values[(square_num*11)+k-1] = 1
            square_num += 1

# ----------------Creation of the input Matrix (normalization)-------------------

def input_matrix_creation_norm(A, Values_2) :
    square_num = 0
    for i in range(1, 5) :
        for j in range(1, 5) :
            for k in range(1, 12) :
                if A[i, j] == 2**k :
                    Values_2[square_num] = k / 11
            square_num += 1

# ---------Shifting pieces over horizontally--------------

def shift_horizontal(A, min_1, min_2, a) :
    i_v2 = min_1
    max_1 = min_1 + 4
    points = 0
    move = False
    merges_done = 0
    while i_v2 < max_1 :
        j_v2 = min_2
        max_2 = min_2 + 4
        while j_v2 < max_2 :
            i = abs(i_v2)
            j = abs(j_v2)
            b = 1
            while A[j, i+a*b] == 0 :
                if A[j, i] != 0 :
                    move = True
                A[j, i+a*b] = A[j, i+a*b-a]
                A[j, i+a*b-a] = 0
                b += 1
            if A[j, i+a*b] == A[j, i+a*b-a] :
                merges_done += 1
                A[j, i+a*b] = A[j, i+a*b] * 2
                points =  points + A[j, i+a*b]
                A[j, i+a*b-a] = 0
                move = True
            j_v2 = j_v2 + 1
        i_v2 = i_v2 + 1
    #    print ("Is there a move", move)
    if move == True :
        two_generator(A)
    return A, move, points, merges_done

# ---------Shifting pieces over vertically--------------

def shift_vertical(A, min_1, min_2, a) :
    i_v2 = min_1
    max_1 = min_1 + 4
    points = 0
    move = False
    merges_done = 0
    while i_v2 < max_1 :
        j_v2 = min_2
        max_2 = min_2 + 4
        while j_v2 < max_2 :
            i = abs(i_v2)
            j = abs(j_v2)
            b = 1
            while A[i+a*b, j] == 0 :
                if A[i, j] != 0 :
                    move = True
                A[i+a*b, j] = A[i+a*b-a, j]
                A[i+a*b-a, j] = 0
                b += 1
            if A[i+a*b, j] == A[i+a*b-a, j] :
                merges_done += 1
                A[i+a*b, j] = A[i+a*b, j] * 2
                points = points + A[i+a*b, j]
                A[i+a*b-a, j] = 0
                move = True
            j_v2 = j_v2 + 1
        i_v2 = i_v2 + 1
    #    print ("Is there a move", move)
    if move == True :
        two_generator(A)
    return A, move, points, merges_done

# ------------Highest Number on the board----------------

def highest_square(A) :
    high = 0
    for i in range(1, 5) :
        for j in range(1, 5) :
            if A[i, j] > high :
                high = A[i, j]
    return high

# ---------Movement on the board------------------

def movement(A, direction) :
    print ("This is the direction", direction)
    if direction == 1 :  # Left
        A, move, points, merges_done = shift_horizontal(A, 1, 1, -1)
        Done = test_done(A)
    if direction == 2 :  # Right
        A, move, points, merges_done = shift_horizontal(A, -4, 1, 1)
        Done = test_done(A)
    if direction == 3 :  # Up
        A, move, points, merges_done = shift_vertical(A, 1, 1, -1)
        Done = test_done(A)
    if direction == 4 :  # Down
        A, move, points, merges_done = shift_vertical(A, -4, 1, 1)
        Done = test_done(A)

    return A, points, Done


# -------------Display the game in Pygame------------------

def display_game(A, total_points, gen, High_Score, decay_rate, Highest_number_seen) :
    
    screen.fill(WHITE)
    # Draw on the screen a line from (0,0) to (100,100)
    # 5 pixels wide.
    for i in range(1, 4) :
        pygame.draw.line(screen, RED, [0, 200*i], [800, 200*i], 5)
        pygame.draw.line(screen, RED, [200*i, 0], [200*i, 800], 5)
    font = pygame.font.SysFont('Calibri', 75, True, False)
    font_2 = pygame.font.SysFont('Calibri', 25, True, False)
    for i in range(0, 4) :
        for j in range(0, 4) :
            text = font.render(str(int(A[i+1, j+1])), True, BLACK)
            screen.blit(text, [80+(200*j), 70+(200*i)])
    gen = font_2.render("Generation: " + str(int(gen)), True, BLACK)
    Present_score = font_2.render("Score: " + str(int(total_points)), True, BLACK)
    Score = font_2.render("High Score: " + str(int(High_Score)), True, BLACK)
    Decay = font_2.render("Deacy Rate: " + str(float(decay_rate)), True, BLACK)
    Big_Num = font_2.render("Biggest Number: " + str(float(Highest_number_seen)), True, BLACK)
    screen.blit(gen, [825, 25])
    screen.blit(Big_Num, [825, 150])
    screen.blit(Score, [825, 300])
    screen.blit(Present_score, [825, 450])
    screen.blit(Decay, [825, 600])

# --------------- Weighted Random Pick ------------------------

def weighted_pick(array) :
    x = random.uniform(0, 1)
    cum_prob = 0.0
    for item in range(4) :
        cum_prob += array[item]
        if x < cum_prob :
            break
    return item


####################################
#### ----------MAIN-------------####
####################################

# ---------------Initialize Pygame-------------------
pygame.init()

# ----------------Important Variables------------------

Num_of_generations = 100000
start_size = 176
end_size = 4
hidden_size = 100
parameters = 4
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
points = 0
new_x = 0
new_y = 0
High_Score = 0
epsilon = 0.5
epsilon_v2 = 0.4
decay_rate_v1 = 0.9995
decay_rate_v2 = 0.99
y = 0.95
size = (1200, 800)
High_Score = 0
Highest_number_seen = 0
smart = False
gen = 0
drop = 0.1
drop_rate = 1.0001

# ----------------Initiate the Pygame Environnement----------------

screen = pygame.display.set_mode(size)
pygame.display.set_caption("2048")
clock = pygame.time.Clock()
Display_A = np.zeros((6, 6))



# ----------------Neural Network-----------------------

model = Sequential()
model.add(Dense(hidden_size*2, input_dim=start_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(end_size))
model.compile(loss='mse', optimizer = 'adam', metrics = ['accuracy'])

# -----------------Let it Evolve-----------------------

while not smart :

    # FPS for pygame
    clock.tick(10)

    # Total Score for this generation
    total_score = 0

    # Display grid
    for i in range(1, 4) :
        pygame.draw.line(screen, RED, [0, 200*i], [800, 200*i], 5)
        pygame.draw.line(screen, RED, [200*i, 0], [200*i, 800], 5)

    A = np.zeros((6, 6))
    B = np.zeros(1)

    # Populate the Array A with 0's to initialize the game.
    for i in range(1, 5) :
        for j in range(1, 5) :
            A[i, j] = 0

    # Make boundaries 1        
    for i in range(0, 6) :
        A[i, 0] = 1
        A[i, 5] = 1
        A[0, i] = 1
        A[5, i] = 1

    # Allows to randomly add a 2 at a random location on the grid. 
    rand_x = random.randint(1, 4)
    rand_y = random.randint(1, 4)
    A[rand_x, rand_y] = 2

    # Variable for this generation
    done = False
    finished = False
    total_points = 0
    epsilon *= decay_rate_v1
    drop *= drop_rate

    while not done:
        screen.fill(WHITE)
        font = pygame.font.SysFont('Calibri', 75, True, False)
            
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                finished = True  # Flag that we are done so we exit this loo
        
        Display_A = A
        Values = np.zeros(start_size)
        Old_A = np.zeros((6, 6))
        input_matrix_creation(A, Values) # Start Inputs
#        input_matrix_creation_norm(A, Values)

        # --------------Choose which Epsilon to use------------------
        
        # Get the highest number in the game
        Fattest_num = highest_square(A)

        if Highest_number_seen < Fattest_num :
            Highest_number_seen = Fattest_num
#            epsilon_v2 = 0.4
#        if Highest_number_seen == Fattest_num :
#            epsilon_v2 *= decay_rate_v2
#            epsilon = epsilon_v2
#        else :
#            epsilon = epsilon_v1
#        try :
#            epsilon = (total_points/High_Score) * 0.25
#        except ZeroDivisionError :
#            epsilon = 0.25
#        print ("This is the epsilon :", epsilon)

        # If random number is lower than the epsilon, the model will do a random move to learn from mistake. Move
        # is represented by 1 - 4. Else, use the model to predict the next move. 
        if np.random.rand() <= epsilon :
            action = np.random.randint(0, end_size, size = 1)
        else :
            q = model.predict(Values.reshape(-1, start_size))

            action = np.argmax(q[0])
#        q = model.predict(Values.reshape(-1, start_size))
#        print ("\n\nthis is the first matrix:", q)
#        print ("\n\nthis is the second matrix", q[0])
#        total_Q_sum = 0
#        bayessian_q = np.zeros(4)
#        for i in range(4) :
#            if q[0][i] < 0 :
#                q[0][i] = 0
#            else :
#                total_Q_sum += q[0][i]
#
#        print (total_Q_sum)
#
#        for i in range(4) :
#            bayessian_q[i] = q[0][i] / total_Q_sum
#        print ("\n\n This is the bayessian matrix", bayessian_q)
#        action = weighted_pick(bayessian_q)
        print ("This is the action number:", action)
        Old_A[:, :] = A[:, :]
        
        New_A, points, done = movement(A, action+1)
        New_Values = np.zeros(start_size)
        input_matrix_creation(New_A, New_Values)
        print ("new Values")
        print (New_Values)
#        input_matrix_creation_norm(New_A, New_Values)

        # ---------------Train the Neural Net by updating it--------------------
        
        add = np.max(model.predict(New_Values.reshape(-1, start_size)))
        print ("This is the addition:", add)
        print ("This is the points:", points)
        target = points + y * add
        target_vec = model.predict(Values.reshape(-1, start_size))[0]
        print(target_vec)
        target_vec[action] = target
        print(target_vec)
        model.train_on_batch(Values.reshape(-1, start_size), target_vec.reshape(-1, end_size))
        
        
        if np.array_equal(Old_A, New_A) :
            done = True
        print ("Is it Done? :", done)
        A = New_A
        total_points += points
        if High_Score < total_points :
            High_Score = total_points
        display_game(A, total_points, gen, High_Score, epsilon, Highest_number_seen)
        pygame.display.update()
        if Highest_number_seen == 2048 :
            smart = True

    gen += 1
model.save_weights("model.h5", overwrite=True)
with open("model.json", "w") as outfile:
    json.dump(model.to_json(), outfile)




