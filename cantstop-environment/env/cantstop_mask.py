import random 

roll_perm = [[0,1,2,3], [0,2,1,3], [0,3,1,2]]

class CantStopActionMask:

    def __init__(self, dicerolls, temp_board): #dicerolls should be a list of four numbers, 1-6, which will be used to determine possible combinations
                                            #temp_board will be used to determine what roles could be used with the current board, and should be an 12-length array
                                            # Adding extra length to temp_board to make indexing easier 
        #print("temp_board", temp_board)
        temp_placed = sum(i >= 0 for i in temp_board[1:]) #count number of temporary pieces placed which will affect what actions will be masked
        
        self.mask = [0]*144
        raw_tots = []
        for i in roll_perm:
                #sort the total for the rolls to reduce the decision space - make biggest number first
                #this also simplifies making the mask, as we can just set the same choice to 1 multiple times
                #subtracting 1 will map 2-12 to 1-11
                temp_tots = [dicerolls[i[0]]+ dicerolls[i[1]] - 1, dicerolls[i[2]] + dicerolls[i[3]] - 1]
                temp_tots.sort()
                    
                raw_tots += [temp_tots]

        if temp_placed <= 1:
            for i in raw_tots:
                #print(i)
                self.mask[i[0]*12+i[1]] = 1 #all rolls are possible

        elif temp_placed == 2:
            for i in raw_tots:
                #if one or the other is on the temp board , can handle both on one because both can always be selected
                if temp_board[i[0]] >= 0 or temp_board[i[1]] >= 0:
                    self.mask[i[0]*12 + i[1]] = 1
                    
                #if neither are on the temp board only one or the other can be added
                else:
                    #handle case where you roll the same value twice
                    if i[0] == i[1]:
                        self.mask[i[0]*12+i[1]] = 1
                    else:
                        self.mask[i[0]] = 1
                        self.mask[i[1]] = 1

        else: #should always be 3
            for i in raw_tots:
                if temp_board[i[0]] >= 0 and temp_board[i[1]] >= 0:
                    self.mask[i[0]*12+i[1]] = 1
                elif temp_board[i[0]] >= 0:
                    self.mask[i[0]] = 1
                elif temp_board[i[1]] >= 1:
                    self.mask[i[1]] = 1

        if sum(self.mask) == 0:
            print("No Possible Move dice rolls: ", dicerolls, " temp board: ", temp_board)
            self.mask[0]=1 #set action for no other possible action


    def __str__(self):
        return str(self.mask)


"""
Simple Test cases

temp_board = [-1]*12
dicerolls = [random.randint(1,6) for i in range(4)]

dicerolls = [5,6,5,6]

temp_board[6] = 0
temp_board[7] = 0

m = CantStopActionMask(dicerolls, temp_board)

print([i for i in range(len(m.mask)) if m.mask[i] > 0])

print(m)

"""