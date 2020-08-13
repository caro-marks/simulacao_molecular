import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import keras 
from keras.models import Sequential
from keras.layers import Dense
import tensorflow


L_list = [10,15,20,25,30]
rho_list = [0.12,0.16,0.22,0.28]

list_of_files = []
for L_aux in (L_list):
    for rho_aux in (rho_list):
        filename_aux = '/mnt/2cdd110c-3460-4e1a-973a-c2f67960125f/vini_HDD/pesquisa/AI_learns_MC/new_ml_data/data__epsilon_infty_L_'+str(L_aux)+'_T_0.25_rho_'+str('{:4.2f}'.format(rho_aux))+'_.dat'
        list_of_files.append(filename_aux)

print(len(list_of_files))


'''
list_of_files = [
    '/mnt/2cdd110c-3460-4e1a-973a-c2f67960125f/vini_HDD/pesquisa/AI_learns_MC/new_ml_data/data__epsilon_infty_L_10_T_0.25_rho_0.10_.dat',
    '/mnt/2cdd110c-3460-4e1a-973a-c2f67960125f/vini_HDD/pesquisa/AI_learns_MC/new_ml_data/data__epsilon_infty_L_10_T_0.25_rho_0.14_.dat',
    '/mnt/2cdd110c-3460-4e1a-973a-c2f67960125f/vini_HDD/pesquisa/AI_learns_MC/new_ml_data/data__epsilon_infty_L_10_T_0.25_rho_0.18_.dat',
    '/mnt/2cdd110c-3460-4e1a-973a-c2f67960125f/vini_HDD/pesquisa/AI_learns_MC/new_ml_data/data__epsilon_infty_L_10_T_0.25_rho_0.22_.dat',
    '/mnt/2cdd110c-3460-4e1a-973a-c2f67960125f/vini_HDD/pesquisa/AI_learns_MC/new_ml_data/data__epsilon_infty_L_10_T_0.25_rho_0.26_.dat',
    '/mnt/2cdd110c-3460-4e1a-973a-c2f67960125f/vini_HDD/pesquisa/AI_learns_MC/new_ml_data/data__epsilon_infty_L_10_T_0.25_rho_0.30_.dat'   ]
'''


#%%




X = []
y = []

for filename in (list_of_files):
    dataset = pd.read_csv(filename, sep = '\s+', header = None)
    X_aux = dataset.iloc[:, 2:10].values
    y_aux = dataset.iloc[:,9].values
    
    how_many_ones = 0
    how_many_zeros = 0
    for i in range(len(X_aux)):
        if(X_aux[i][7] == 1):
            how_many_ones+= 1
        else:
            how_many_zeros += 1
    
    print(how_many_ones, how_many_zeros)
    one_count = 0
    zero_count = 0
    
    for i in range(len(X_aux)):
        if(X_aux[i][7] == 1):
            if(one_count <= how_many_ones):
                X.append(X_aux[i])
                #y.append(y_aux[i])
                one_count += 1
        else:
            if(zero_count <= how_many_ones):
                X.append(X_aux[i])
                #y.append(y_aux[i])
                zero_count += 1
        
        
        
        
how_many_ones = 0
how_many_zeros = 0
for i in range(len(X)):
    if(X[i][7] == 1):
        how_many_ones+= 1
    else:
        how_many_zeros += 1
print('#############')
print(how_many_ones, how_many_zeros)

#%%
zerones_diff = abs(how_many_ones-how_many_zeros)
if(zerones_diff !=0):
    #print(zerones_diff) 
    if(how_many_ones > how_many_zeros):
        value_to_purge = 1
    else:
        value_to_purge = 0
    this_much_have_been_purged = 0
    while(this_much_have_been_purged < zerones_diff):
        print(this_much_have_been_purged)
        for i in range(len(X)):
            if(X[i][7] == value_to_purge):
                #y.pop(i)
                X.pop(i)
                break
        this_much_have_been_purged += 1
        
#%%
        
how_many_ones = 0
how_many_zeros = 0
for i in range(len(X)):
    if(X[i][7] == 1):
        how_many_ones+= 1
    else:
        how_many_zeros += 1
print('#############')
print(how_many_ones, how_many_zeros)
print('#############')
print(len(X[0]), len(X),how_many_ones+how_many_zeros) 

#print(X[0][0], y[0])

#print(len(X[0]))

print(X[0])

#%%
X2 = np.zeros((len(X), 7))
y2 = np.zeros((len(X), 1))
for i in range(len(X)):
    for j in range(len(X[i]) - 1):
        X2[i,j] = X[i][j]
    y2[i] = X[i][7]

#print(X2)

#%%
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.2, random_state = 666)

print(X_train[0])
classifier = Sequential()

classifier.add(Dense(units = 12, init = 'uniform', activation = 'relu', input_dim = 7))

#classifier.add(Dense(units = 24, init = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 12, init = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#%%

classifier.fit(X_train, y_train, batch_size = 128, nb_epoch = 10)

#%%

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

print('\n')

from sklearn.metrics import accuracy_score
a = accuracy_score(y_pred,y_test)
print('Accuracy is:', a*100)

#%%

testizinho = np.zeros((7,1))
for i in range(7):
    testizinho[i] = 0
testizinho[0] = 0
testizinho[1] = 1
testizinho[2] = 1
testizinho = testizinho.transpose()
print(testizinho)
one_pred = classifier.predict(testizinho)

print(one_pred)

#%%




def mcmove(sites, respulsion_factor, T):

        # ---- randomly pick one occupied site ----
    occupied_site_pick = False
    while(not occupied_site_pick):
        random_pick = [np.random.randint(0,L), np.random.randint(0,L)]
        if(sites[random_pick[0]][random_pick[1]][2] == 1):
            occupied_site_pick = True

    move_particle = False
    random_neighbor = np.random.randint(0,6)    # randomly choose one neighbor from neighbor-0 to neighbor-5

    if(random_neighbor == 0):      # ngb 0 -> bottom left neighbor
        if(random_pick[0] % 2 == 0):
            i = random_pick[0] - 1
            j = random_pick[1] - 1
        else:
            i = random_pick[0] - 1
            j = random_pick[1]

    elif(random_neighbor == 1):    # ngb 1 -> left neighbor   
        i = random_pick[0]
        j = random_pick[1] - 1

    elif(random_neighbor == 2):    # ngb 2 -> top left neighbor
        if(random_pick[0] % 2 == 0):
            i = random_pick[0] + 1
            j = random_pick[1] - 1
        else:
            i = random_pick[0] + 1
            j = random_pick[1]

    elif(random_neighbor == 3):    # ngb 3 -> top right neighbor
        if(random_pick[0] % 2 == 0):
            i = random_pick[0] + 1
            j = random_pick[1]
        else:
            i = random_pick[0] + 1
            j = random_pick[1] + 1

    elif(random_neighbor == 4):    # ngb 4 -> right neighbor
        i = random_pick[0]
        j = random_pick[1] + 1

    elif(random_neighbor == 5):    # ngb 5 -> bottom right neighbor
        if(random_pick[0] % 2 == 0):
            i = random_pick[0] - 1
            j = random_pick[1]
        else:
            i = random_pick[0] - 1
            j = random_pick[1] + 1

    if(i == L): i = 0     # the right/upper ngb of the last column/row is the first site of next column/row
    if(j == L): j = 0


    ngb_site_is_free = False
    free_neighbor_picked = [0,0]
    
    if(sites[i][j][2] == 0):
        ngb_site_is_free = True
        free_neighbor_picked = [i, j] 
    else:
        free_neighbor_picked = [i, j]

    if(ngb_site_is_free):

        if(repulsion_factor == 0):
        
            nj = [0,0,0,0,0,0]
            
            row_above = free_neighbor_picked[0] + 1
            if(row_above == L): row_above = 0
            row_below = free_neighbor_picked[0] - 1
            same_row = free_neighbor_picked[0]
            right_col = free_neighbor_picked[1] + 1
            if(right_col == L): right_col = 0
            left_col = free_neighbor_picked[1] - 1
            if(free_neighbor_picked[0] % 2 == 0):
                diff_row_right_col = free_neighbor_picked[1]
                diff_row_left_col = free_neighbor_picked[1] - 1
            else:
                diff_row_right_col = free_neighbor_picked[1] + 1
                if(diff_row_right_col == L): diff_row_right_col = 0
                diff_row_left_col = free_neighbor_picked[1]


            nj[0] = sites[row_below][diff_row_left_col][2]    # bottom left neighbor
            nj[1] = sites[row_above][diff_row_left_col][2]    # top left neighbor
            nj[2] = sites[row_below][diff_row_right_col][2]   # bottom right neighbor
            nj[3] = sites[row_above][diff_row_right_col][2]   # top right neighbor
            nj[4] = sites[same_row][left_col][2]              # left neighbor
            nj[5] = sites[same_row][right_col][2]             # right neighbor

            acc = 0
            for k in range(number_of_ngbs):
                acc += nj[k]
            
            if(acc == 1):
                move_particle = True


        else:

            energy_before_moving = energy_calculation(sites, repulsion_factor)
            
            sites[random_pick[0]][random_pick[1]][2] = 0
            sites[free_neighbor_picked[0]][free_neighbor_picked[1]][2] = 1

            energy_after_moving = energy_calculation(sites, repulsion_factor)

            de = energy_after_moving - energy_before_moving
            
            r = np.random.random()
            if(de < 0 or r < np.exp(-de/T)):
                move_particle = True
            
            sites[random_pick[0]][random_pick[1]][2] = 1
            sites[free_neighbor_picked[0]][free_neighbor_picked[1]][2] = 0

    else:
        move_particle = False


    return move_particle, ngb_site_is_free, random_pick[0], random_pick[1], free_neighbor_picked[0], free_neighbor_picked[1]    


#######################################
#      Main()
#######################################


np.random.seed(4)            # random seed
number_of_MC_steps = 100000    # number of Monte Carlo steps to be performed after equilibration
number_of_equil_steps = 10000            # number of steps for equilibrate the system
#L = 8                        # lenght of the box's side (box -> L*L)
number_of_ngbs = 6           # number of neighbors for each site = 6 for an hexagonal / triangular lattice
#T = 2                    # temperature

#repulsion_factor = 10   # epsilon_{i,j} in the Hamiltonian of the system - if zero -> infinite-repulsion case
#occupation_density = 0.5   


ff = open('acceptance_rate_PREDICTED.dat', 'w+')
for repulsion_factor in ([0]):
    for L in ([10,15,20,25,30]):
        for T in ([0.25]):
            for occupation_density in ([0.1,0.15,0.20,0.25,0.3]):
                
                number_of_sites_occupied = int(L*L*occupation_density)
                sites = [[[0,0,0] for i in range(L)] for j in range(L)]     # allocate array for each site [position_x, position_y, occupation_state]

                particle_id_and_pos = []

                # ---- generate hexagonal lattice with all sites unoccupied ----
                k = 0
                l = 0
                for i in range(L):
                    for j in range(L):
                        if(i % 2 == 0):
                            sites[i][j][0] = 2*j
                        else:
                            sites[i][j][0] = 2*j + 1
                        sites[i][j][1] = 2*k
                        particle_id_and_pos.append([l, sites[i][j][0], sites[i][j][1]])
                        l += 1
                    k += 1


                # ---- randomly occupy sites, without overlaps ----
                number_of_allocation_tentatives = 0
                overlap = True
                while(overlap):
                    number_of_allocation_tentatives += 1
                    if(number_of_allocation_tentatives == 100000):
                        #print('#######################')
                        #print('NUMBER OF TENTATIVES TO ALLOCATE SYSTEM REACHED MAX FOR SYSTEM:')
                        #print('eps: '+str(repulsion_factor)+' L: '+str(L)+' T: '+str(T)+' rho: '+str(occupation_density))
                        break
                    random_positions_array = []
                    for i in range(number_of_sites_occupied):
                        random_positions_array.append([np.random.randint(0,L),np.random.randint(0,L)])
                    overlap = False
                    for i in range(number_of_sites_occupied):
                        for j in range(number_of_sites_occupied):
                            if(i != j):
                                if(random_positions_array[i] == random_positions_array[j]):
                                    overlap = True
                #if(number_of_allocation_tentatives == 100000):
                    #print('#######################')
                    #break
                for i in range(number_of_sites_occupied):
                    sites[random_positions_array[i][0]][random_positions_array[i][1]][2] = 1


                
                '''
                # ---- show lattice with occupied/ unoccupied sites ---- 
                for i in range(L):
                    for j in range(L):
                        if(sites[i][j][2] == 0):
                            color = 'b'
                            size = 100
                        else:
                            color = 'r'
                            size = 1000
                        plt.scatter(sites[i][j][0], sites[i][j][1], c = color, s = size)
                plt.savefig('lattice.png')
                #plt.show()
                plt.close()
                '''
                


                
                if(repulsion_factor == 0):
                    epsilon = 'infty'
                else:
                    epsilon = str(repulsion_factor)
                
                f = open('void1.dat', 'w+')#'./results_cnn/snap__epsilon_'+epsilon+'_L_'+str('{:02d}'.format(L))+'_T_'+str('{:4.2f}'.format(T))+'_rho_'+str('{:4.2f}'.format(occupation_density))+'_PREDICTED.xyz', 'w+')
                g = open('void2.dat', 'w+')#./results_cnn/msd__epsilon_'+epsilon+'_L_'+str('{:02d}'.format(L))+'_T_'+str('{:4.2f}'.format(T))+'_rho_'+str('{:4.2f}'.format(occupation_density))+'_PREDICTED.dat', 'w+')
                #h = open('./new_ml_data/data__epsilon_'+epsilon+'_L_'+str('{:02d}'.format(L))+'_T_'+str('{:4.2f}'.format(T))+'_rho_'+str('{:4.2f}'.format(occupation_density))+'_.dat', 'w+')
                #print('_'+epsilon+'_L_'+str('{:02d}'.format(L))+'_T_'+str('{:4.2f}'.format(T))+'_rho_'+str('{:4.2f}'.format(occupation_density)))
                
                for eq_step in range(number_of_equil_steps):

                    move_particle, free_ngb, row_original_site, col_original_site, row_new_site, col_new_site = mcmove(sites, repulsion_factor, T)

                    if(move_particle):
                        sites[row_original_site][col_original_site][2] = 0
                        sites[row_new_site][col_new_site][2] = 1



                acceptance_rate = 0
                disp = 0
                for MC_step in range(number_of_MC_steps): 

                    move_particle, free_ngb, row_original_site, col_original_site, row_new_site, col_new_site = mcmove(sites, repulsion_factor, T)

                    ni = sites[row_new_site][col_new_site][2]
                    nj = [0,0,0,0,0,0]

                    row_above = row_new_site + 1
                    if(row_above == L): row_above = 0
                    row_below = row_new_site - 1
                    same_row = row_new_site
                    right_col = col_new_site + 1
                    if(right_col == L): right_col = 0
                    left_col = col_new_site - 1
                    if(row_new_site % 2 == 0):
                        diff_row_right_col = col_new_site
                        diff_row_left_col = col_new_site - 1
                    else:
                        diff_row_right_col = col_new_site + 1
                        if(diff_row_right_col == L): diff_row_right_col = 0
                        diff_row_left_col = col_new_site


                    nj[0] = sites[row_below][diff_row_left_col][2]    # bottom left neighbor
                    nj[1] = sites[row_above][diff_row_left_col][2]    # top left neighbor
                    nj[2] = sites[row_below][diff_row_right_col][2]   # bottom right neighbor
                    nj[3] = sites[row_above][diff_row_right_col][2]   # top right neighbor
                    nj[4] = sites[same_row][left_col][2]              # left neighbor
                    nj[5] = sites[same_row][right_col][2]             # right neighbor

                
                    nj2 = np.zeros((7,1))
                    nj2[0] = ni
                    for counter in range(1,7):
                        nj2[counter] = nj[counter-1]
                
                    nj2 = nj2.transpose()
                
                    mexeu = classifier.predict(nj2)
                    if(mexeu > 0.5):
                        move_particle = True
                    else:
                        move_particle = False
                    
                    '''
                    plt.scatter(sites[row_below][diff_row_left_col][0], sites[row_below][diff_row_left_col][1], c = 'b')
                    plt.scatter(sites[row_above][diff_row_left_col][0], sites[row_above][diff_row_left_col][1], c = 'b')
                    plt.scatter(sites[row_below][diff_row_right_col][0], sites[row_below][diff_row_right_col][1], c = 'b')
                    plt.scatter(sites[row_above][diff_row_right_col][0], sites[row_above][diff_row_right_col][1], c = 'b')
                    plt.scatter(sites[same_row][left_col][0], sites[same_row][left_col][1], c = 'b')
                    plt.scatter(sites[same_row][right_col][0], sites[same_row][right_col][1], c = 'b')
                    plt.scatter(sites[row_new_site][col_new_site][0], sites[row_new_site][col_new_site][1], c = 'g')
                    plt.show()
                    plt.close()
                    '''

                    if(move_particle):
                        #mexeu = 1
                        #print('moved')
                        acceptance_rate += 1
                    #else:
                    #mexeu = 0

                    '''
                    if(free_ngb):
                        vizin = 0
                    else:
                        vizin = 1
                    '''

                    #h.write(str(repulsion_factor)+'  '+str(T)+'  '+str(vizin)+'  '+str(nj[0])+'  '+str(nj[1])+'  '+str(nj[2])+'  '+str(nj[3])+'  '+str(nj[4])+'  '+str(nj[5])+ '    ' + str(mexeu) + '\n')

                    if(move_particle):
                        sites[row_original_site][col_original_site][2] = 0
                        sites[row_new_site][col_new_site][2] = 1
                        f.write(str(L*L) + '\n')
                        f.write('PysingNN' + '\n')
                        for i in range(L):
                            for j in range(L):
                                if(sites[i][j][2] == 0):
                                    occupation_state = 'O'
                                else:
                                    occupation_state = 'F'
                                f.write(occupation_state + '    '+ str(sites[i][j][0]) + '    ' + str(sites[i][j][1]) + '\n')

                    if(move_particle): disp += 1
                    msd = disp
                    
                    g.write(str(MC_step) + '    ' + str(msd) + '\n')
                
                acceptance_rate /= number_of_MC_steps
                ff.write(str(repulsion_factor)+'  '+str('{:5.3f}'.format(L))+'  '+str('{:5.3f}'.format(T))+'  '+str('{:5.3f}'.format(occupation_density))+'  '+str('{:5.3f}'.format(acceptance_rate)) + '\n')
                print(str(repulsion_factor)+'  '+str('{:5.3f}'.format(L))+'  '+str('{:5.3f}'.format(T))+'  '+str('{:5.3f}'.format(occupation_density))+'  '+str('{:5.3f}'.format(acceptance_rate)) + '\n')











































































#%%

######################################################
###                                                ###
###     NEXT BLOCK CONTAINS OLDER MC VERSION       ###
###     I DIDNT WANTED TO DELETE, SOOOOOOOOO       ###
###                                                ###
######################################################

#%%
def mcmove(sites, respulsion_factor, T):

        # ---- randomly pick one occupied site ----
    occupied_site_pick = False
    while(not occupied_site_pick):
        random_pick = [np.random.randint(0,L), np.random.randint(0,L)]
        if(sites[random_pick[0]][random_pick[1]][2] == 1):
            occupied_site_pick = True

    move_particle = False
    random_neighbor = np.random.randint(0,6)    # randomly choose one neighbor from neighbor-0 to neighbor-5

    if(random_neighbor == 0):      # ngb 0 -> bottom left neighbor
        if(random_pick[0] % 2 == 0):
            i = random_pick[0] - 1
            j = random_pick[1] - 1
        else:
            i = random_pick[0] - 1
            j = random_pick[1]

    elif(random_neighbor == 1):    # ngb 1 -> left neighbor   
        i = random_pick[0]
        j = random_pick[1] - 1

    elif(random_neighbor == 2):    # ngb 2 -> top left neighbor
        if(random_pick[0] % 2 == 0):
            i = random_pick[0] + 1
            j = random_pick[1] - 1
        else:
            i = random_pick[0] + 1
            j = random_pick[1]

    elif(random_neighbor == 3):    # ngb 3 -> top right neighbor
        if(random_pick[0] % 2 == 0):
            i = random_pick[0] + 1
            j = random_pick[1]
        else:
            i = random_pick[0] + 1
            j = random_pick[1] + 1

    elif(random_neighbor == 4):    # ngb 4 -> right neighbor
        i = random_pick[0]
        j = random_pick[1] + 1

    elif(random_neighbor == 5):    # ngb 5 -> bottom right neighbor
        if(random_pick[0] % 2 == 0):
            i = random_pick[0] - 1
            j = random_pick[1]
        else:
            i = random_pick[0] - 1
            j = random_pick[1] + 1

    if(i == L): i = 0     # the right/upper ngb of the last column/row is the first site of next column/row
    if(j == L): j = 0


    ngb_site_is_free = False
    free_neighbor_picked = [0,0]
    
    if(sites[i][j][2] == 0):
        ngb_site_is_free = True
        free_neighbor_picked = [i, j] 
    
    if(ngb_site_is_free):

        if(repulsion_factor == 0):
        
            nj = [0,0,0,0,0,0]
            
            row_above = free_neighbor_picked[0] + 1
            if(row_above == L): row_above = 0
            row_below = free_neighbor_picked[0] - 1
            same_row = free_neighbor_picked[0]
            right_col = free_neighbor_picked[1] + 1
            if(right_col == L): right_col = 0
            left_col = free_neighbor_picked[1] - 1
            if(free_neighbor_picked[0] % 2 == 0):
                diff_row_right_col = free_neighbor_picked[1]
                diff_row_left_col = free_neighbor_picked[1] - 1
            else:
                diff_row_right_col = free_neighbor_picked[1] + 1
                if(diff_row_right_col == L): diff_row_right_col = 0
                diff_row_left_col = free_neighbor_picked[1]


            nj[0] = sites[row_below][diff_row_left_col][2]    # bottom left neighbor
            nj[1] = sites[row_above][diff_row_left_col][2]    # top left neighbor
            nj[2] = sites[row_below][diff_row_right_col][2]   # bottom right neighbor
            nj[3] = sites[row_above][diff_row_right_col][2]   # top right neighbor
            nj[4] = sites[same_row][left_col][2]              # left neighbor
            nj[5] = sites[same_row][right_col][2]             # right neighbor

            acc = 0
            for k in range(number_of_ngbs):
                acc += nj[k]
            
            if(acc == 1):
                move_particle = True


        else:

            energy_before_moving = energy_calculation(sites, repulsion_factor)
            
            sites[random_pick[0]][random_pick[1]][2] = 0
            sites[free_neighbor_picked[0]][free_neighbor_picked[1]][2] = 1

            energy_after_moving = energy_calculation(sites, repulsion_factor)

            de = energy_after_moving - energy_before_moving
            
            r = np.random.random()
            if(de < 0 or r < np.exp(-de/T)):
                move_particle = True
            
            sites[random_pick[0]][random_pick[1]][2] = 1
            sites[free_neighbor_picked[0]][free_neighbor_picked[1]][2] = 0

    return move_particle, random_pick[0], random_pick[1], free_neighbor_picked[0], free_neighbor_picked[1]    




















np.random.seed(666)            # random seed
number_of_MC_steps = 20000    # number of Monte Carlo steps to be performed after equilibration
number_of_equil_steps = 200            # number of steps for equilibrate the system
#L = 8                        # lenght of the box's side (box -> L*L)
number_of_ngbs = 6           # number of neighbors for each site = 6 for an hexagonal / triangular lattice
#T = 2                    # temperature

#repulsion_factor = 10   # epsilon_{i,j} in the Hamiltonian of the system - if zero -> infinite-repulsion case
#occupation_density = 0.5   

L = 12
T = 0.25
occupation_density = 0.33
repulsion_factor = 0



number_of_sites_occupied = int(L*L*occupation_density)
sites = [[[0,0,0] for i in range(L)] for j in range(L)]     # allocate array for each site [position_x, position_y, occupation_state]

particle_id_and_pos = []

# ---- generate hexagonal lattice with all sites unoccupied ----
k = 0
l = 0
for i in range(L):
    for j in range(L):
        if(i % 2 == 0):
            sites[i][j][0] = 2*j
        else:
            sites[i][j][0] = 2*j + 1
        sites[i][j][1] = 2*k
        particle_id_and_pos.append([l, sites[i][j][0], sites[i][j][1]])
        l += 1
    k += 1


# ---- randomly occupy sites, without overlaps ----
number_of_allocation_tentatives = 0
overlap = True
while(overlap):
    number_of_allocation_tentatives += 1
    if(number_of_allocation_tentatives == 100000):
        print('#######################')
        print('NUMBER OF TENTATIVES TO ALLOCATE SYSTEM REACHED MAX FOR SYSTEM:')
        print('eps: '+str(repulsion_factor)+' L: '+str(L)+' T: '+str(T)+' rho: '+str(occupation_density))
        break
    random_positions_array = []
    for i in range(number_of_sites_occupied):
        random_positions_array.append([np.random.randint(0,L),np.random.randint(0,L)])
    overlap = False
    for i in range(number_of_sites_occupied):
        for j in range(number_of_sites_occupied):
            if(i != j):
                if(random_positions_array[i] == random_positions_array[j]):
                    overlap = True
'''if(number_of_allocation_tentatives == 100000):
    print('#######################')
    break'''
for i in range(number_of_sites_occupied):
    sites[random_positions_array[i][0]][random_positions_array[i][1]][2] = 1



'''
# ---- show lattice with occupied/ unoccupied sites ---- 
for i in range(L):
    for j in range(L):
        if(sites[i][j][2] == 0):
            color = 'b'
            size = 100
        else:
            color = 'r'
            size = 1000
        plt.scatter(sites[i][j][0], sites[i][j][1], c = color, s = size)
plt.savefig('lattice.png')
#plt.show()
plt.close()
'''




if(repulsion_factor == 0):
    epsilon = 'infty'
else:
    epsilon = str(repulsion_factor)

f = open('./results_cnn/snap__epsilon_'+epsilon+'_L_'+str('{:02d}'.format(L))+'_T_'+str('{:4.2f}'.format(T))+'_rho_'+str('{:4.2f}'.format(occupation_density))+'_PREDICTED.xyz', 'w+')
g = open('./results_cnn/msd__epsilon_'+epsilon+'_L_'+str('{:02d}'.format(L))+'_T_'+str('{:4.2f}'.format(T))+'_rho_'+str('{:4.2f}'.format(occupation_density))+'_PREDICTED.dat', 'w+')
#h = open('./nn_mc/2data__epsilon_'+epsilon+'_L_'+str('{:02d}'.format(L))+'_T_'+str('{:4.2f}'.format(T))+'_rho_'+str('{:4.2f}'.format(occupation_density))+'_.dat', 'w+')
print('_'+epsilon+'_L_'+str('{:02d}'.format(L))+'_T_'+str('{:4.2f}'.format(T))+'_rho_'+str('{:4.2f}'.format(occupation_density)))

for eq_step in range(number_of_equil_steps):

    move_particle, row_original_site, col_original_site, row_new_site, col_new_site = mcmove(sites, repulsion_factor, T)

    if(move_particle):
        sites[row_original_site][col_original_site][2] = 0
        sites[row_new_site][col_new_site][2] = 1

disp = 0
for MC_step in range(number_of_MC_steps): 

    move_particle, row_original_site, col_original_site, row_new_site, col_new_site = mcmove(sites, repulsion_factor, T)

    ni = sites[row_new_site][col_new_site][2]
    nj = [0,0,0,0,0,0]

    row_above = row_new_site + 1
    if(row_above == L): row_above = 0
    row_below = row_new_site - 1
    same_row = row_new_site
    right_col = col_new_site + 1
    if(right_col == L): right_col = 0
    left_col = col_new_site - 1
    if(i % 2 == 0):
        diff_row_right_col = col_new_site
        diff_row_left_col = col_new_site - 1
    else:
        diff_row_right_col = col_new_site + 1
        if(diff_row_right_col == L): diff_row_right_col = 0
        diff_row_left_col = col_new_site

    
    
    nj[0] = sites[row_below][diff_row_left_col][2]    # bottom left neighbor
    nj[1] = sites[row_above][diff_row_left_col][2]    # top left neighbor
    nj[2] = sites[row_below][diff_row_right_col][2]   # bottom right neighbor
    nj[3] = sites[row_above][diff_row_right_col][2]   # top right neighbor
    nj[4] = sites[same_row][left_col][2]              # left neighbor
    nj[5] = sites[same_row][right_col][2]             # right neighbor


    nj2 = np.zeros((7,1))
    nj2[0] = ni
    for counter in range(1,7):
        nj2[counter] = nj[counter-1]

    nj2 = nj2.transpose()

    mexeu = classifier.predict(nj2)
    if(mexeu > 0.5):
        move_particle = True
    else:
        move_particle = False
    
    '''
    if(move_particle):
        mexeu = 1
    else:
        mexeu = 0
    '''
    #h.write(str(repulsion_factor)+'  '+str(T)+'  '+str(ni)+'  '+str(nj[0])+'  '+str(nj[1])+'  '+str(nj[2])+'  '+str(nj[3])+'  '+str(nj[4])+'  '+str(nj[5])+'    ' + str(mexeu)+ '\n')

    if(move_particle):
        sites[row_original_site][col_original_site][2] = 0
        sites[row_new_site][col_new_site][2] = 1
        f.write(str(L*L) + '\n')
        f.write('Pysing' + '\n')
        for i in range(L):
            for j in range(L):
                if(sites[i][j][2] == 0):
                    occupation_state = 'O'
                else:
                    occupation_state = 'F'
                f.write(occupation_state + '    '+ str(sites[i][j][0]) + '    ' + str(sites[i][j][1]) + '\n')

    if(move_particle): disp += 1
    msd = disp
    
    g.write(str(MC_step) + '    ' + str(msd) + '\n')


