import numpy as np
import matplotlib.pyplot as plt 

def energy_calculation(sites, repulsion_factor):
    energy = 0
    
    for i in range(L):
        for j in range(L):
            ni = sites[i][j][2] #center particle
            nj = [0,0,0,0,0,0]

            row_above = i + 1
            if(row_above == L): row_above = 0
            row_below = i - 1
            same_row = i
            right_col = j + 1
            if(right_col == L): right_col = 0
            left_col = j - 1
            if(i % 2 == 0):
                diff_row_right_col = j
                diff_row_left_col = j - 1
            else:
                diff_row_right_col = j + 1
                if(diff_row_right_col == L): diff_row_right_col = 0
                diff_row_left_col = j


            nj[0] = sites[row_below][diff_row_left_col][2]    # bottom left neighbor
            nj[1] = sites[row_above][diff_row_left_col][2]    # top left neighbor
            nj[2] = sites[row_below][diff_row_right_col][2]   # bottom right neighbor
            nj[3] = sites[row_above][diff_row_right_col][2]   # top right neighbor
            nj[4] = sites[same_row][left_col][2]              # left neighbor
            nj[5] = sites[same_row][right_col][2]             # right neighbor

            for k in range(len(nj)):
                energy += ni*nj[k]*repulsion_factor

    energy = 0.5*energy
    return energy


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


ff = open('acceptance_rate.dat', 'w+')
for repulsion_factor in ([0]):
    for L in ([10,15,20,25,30]):
        for T in ([0.25]):
            for occupation_density in ([0.1,0.12,0.14,0.15,0.16,0.18,0.20,0.22,0.24,0.25,0.26,0.28,0.3]):
                
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
                    print(sites, np.array(sites).shape)
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
                if(number_of_allocation_tentatives == 100000):
                    print('#######################')
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
                
                f = open('./new_snapshots/snap__epsilon_'+epsilon+'_L_'+str('{:02d}'.format(L))+'_T_'+str('{:4.2f}'.format(T))+'_rho_'+str('{:4.2f}'.format(occupation_density))+'_.xyz', 'w+')
                g = open('./new_msds/msd__epsilon_'+epsilon+'_L_'+str('{:02d}'.format(L))+'_T_'+str('{:4.2f}'.format(T))+'_rho_'+str('{:4.2f}'.format(occupation_density))+'_.dat', 'w+')
                h = open('./new_ml_data/data__epsilon_'+epsilon+'_L_'+str('{:02d}'.format(L))+'_T_'+str('{:4.2f}'.format(T))+'_rho_'+str('{:4.2f}'.format(occupation_density))+'_.dat', 'w+')
                print('_'+epsilon+'_L_'+str('{:02d}'.format(L))+'_T_'+str('{:4.2f}'.format(T))+'_rho_'+str('{:4.2f}'.format(occupation_density)))
                
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
                        mexeu = 1
                        acceptance_rate += 1
                    else:
                        mexeu = 0

                    if(free_ngb):
                        vizin = 0
                    else:
                        vizin = 1


                    h.write(str(repulsion_factor)+'  '+str(T)+'  '+str(vizin)+'  '+str(nj[0])+'  '+str(nj[1])+'  '+str(nj[2])+'  '+str(nj[3])+'  '+str(nj[4])+'  '+str(nj[5])+ '    ' + str(mexeu) + '\n')

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
                
                acceptance_rate /= number_of_MC_steps
                ff.write(str(repulsion_factor)+'  '+str('{:5.3f}'.format(L))+'  '+str('{:5.3f}'.format(T))+'  '+str('{:5.3f}'.format(occupation_density))+'  '+str('{:5.3f}'.format(acceptance_rate)) + '\n')