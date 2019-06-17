import nashpy as nash
import numpy as np
import pandas as pd
import pickle
from itertools import product
from copy import deepcopy as dcp

class NashEqu():
    def __init__(self, sureWinningRegion, filename = "NashEqu_param.json", ):
        if filename is None:
            print("Error! The file name cannot be Empty!!!")
        json_data = pd.read_json(filename, typ='series')
        self.A = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}
        self.Height = json_data.HEIGHT
        self.Width = json_data.WIDTH
        self.S = self.set_S()
        self.R = self.get_R()
        self.goals = json_data.GOALS
        self.reachGoalReward = json_data.REACHGOALREWARD
        self.catch_reward = json_data.CATCHREWARD
        self.obstacles = json_data.OBSTACLES
        self.epsilon1 = json_data.TRANSITION_PROB1
        self.epsilon2 = json_data.TRANSITION_PROB2
        self.distancethreshold = json_data.DISTANCETHRESHOLD
        self.valuethreshold = 0.0001
        self.decoys = {}
        self.true_goal = []
        self.surewinning = sureWinningRegion
        g = 'g'
        for i in range(len(self.goals)):
            key = g + str(i + 1)
            self.decoys[key] = []
            g_i = tuple([self.goals[i][2], self.goals[i][3]])  ##agent2 认为的goal
            g_ctrl = tuple([self.goals[i][0], self.goals[i][1]])  ##agent1 自己的goal
            self.decoys[key].append(g_i)

            if g_i == g_ctrl:
                self.true_goal.append(g_ctrl)

        self.catch = self.get_catch()
        self.P_s1_a_s2 = self.get_P()
        filename = "P_s1_a_s2.pkl"
        picklefile = open(filename, "wb")
        pickle.dump(self.P_s1_a_s2, picklefile)
        picklefile.close()
        self.actionDict = self.getActionDict()
        self.init_reward_1 = {}
        self.init_reward_2 = {}
        for decoy in self.decoys:
        #     self.init_reward_1[decoy], self.init_reward_2[decoy] = self.init_reward(decoy)
            filename = "reward796.pkl"
            with open(filename, "rb") as f1:
                self.init_reward_1[decoy] = pickle.load(f1)

    def init_reward(self, goal):
        reward_1 = {}
        reward_2 = {}
        terminal = self.decoys[goal]
        for state in self.S:
            # s = tuple(state)
            if state in self.surewinning:
                reward_1[state] = self.reachGoalReward
                reward_2[state] = -self.reachGoalReward
            elif state in self.catch:
                reward_1[state] = -self.catch_reward
                reward_2[state] = self.catch_reward
            else:
                reward_1[state] = 0
                reward_2[state] = 0
        return reward_1, reward_2


    def set_S(self):
        height = self.Height
        width = self.Width
        inner = []
        for p1, q1, p2, q2 in product(range(height), range(width), repeat=2):
            inner.append(((p1, q1), (p2, q2)))
        # print("total number of states: ", len(inner))
        return inner

    def get_R(self):
        R = []
        height = self.Height
        width = self.Width
        for p1, q1 in product(range(height), range(width)):
            R.append((p1, q1))
        return R

    def get_catch(self):
        catch = []
        for state in self.S:
            s = tuple(state)
            if (abs(s[0][0] - s[1][0]) + abs(s[0][1] - s[1][1])) <= self.distancethreshold:
                catch.append(s)
        return catch

    def getActionDict(self):
        actionDict = {}
        actionDict[0] = "N"
        actionDict[1] = "S"
        actionDict[2] = "W"
        actionDict[3] = "E"
        return actionDict

    def trans_P(self, state, id):
        A = self.A
        R = self.R
        P = {}
        P[state] = {}
        st1 = state[0]
        st2 = state[1]
        if id == 1:
            if list(st1) not in self.obstacles and st1 not in self.true_goal and tuple(state) not in self.catch:
                epsilon = self.epsilon1
                explore = []  ##single state
                for action in A:
                    temp = tuple(np.array(st1) + np.array(A[action]))
                    explore.append(temp)
                for action in A:
                    unit1 = epsilon / 4
                    P[state][action] = {}
                    P[state][action][st1] = unit1
                    temp_st1 = tuple(np.array(st1) + np.array(A[action]))
                    if temp_st1 in R and list(temp_st1) not in self.obstacles:
                        # print("temp_st1 is in R:" ,temp_st1)
                        P[state][action][temp_st1] = 1 - epsilon  ## difference is here, large probability to move
                        for _st_ in explore:
                            if _st_ != temp_st1:
                                if _st_ in R:
                                    P[state][action][_st_] = unit1
                                else:
                                    P[state][action][st1] += unit1
                    else:  ##next step will out of range or enter obstacles
                        P[state][action][st1] = 1 - epsilon + unit1  ##difference is here, large probability to remain
                        for _st_ in explore:
                            if _st_ != temp_st1:
                                if _st_ in R:
                                    P[state][action][_st_] = unit1
                                else:
                                    P[state][action][st1] += unit1

            else:  ##agent1 now in obstacle or true goal or been caught, so each action will remain in the obstacle
                for action in A:
                    P[state][action] = {}
                    P[state][action][st1] = 1.0
        else:  ## for agent 2, we dont care about obstacles or goals
            if tuple(state) not in self.catch and st1 not in self.obstacles and st1 not in self.true_goal:
                epsilon = self.epsilon2
                explore = []
                for action in A:
                    temp = tuple(np.array(st2) + np.array(A[action]))
                    explore.append(temp)
                for action in A:
                    unit2 = epsilon / 4
                    P[state][action] = {}
                    P[state][action][st2] = unit2
                    temp_st2 = tuple(np.array(st2) + np.array(A[action]))
                    if temp_st2 in R:
                        P[state][action][temp_st2] = 1 - epsilon
                        for _st_ in explore:
                            if _st_ != temp_st2:
                                if _st_ in R:
                                    P[state][action][_st_] = unit2
                                else:
                                    P[state][action][st2] += unit2
                    else:
                        P[state][action][st2] = 1 - epsilon + unit2
                        for _st_ in explore:
                            if _st_ != temp_st2:
                                if _st_ in R:
                                    P[state][action][_st_] = unit2
                                else:
                                    P[state][action][st2] += unit2
            else:
                for action in A:
                    P[state][action] = {}
                    P[state][action][st2] = 1.0

        return P  ##here P is in the form of P[(st1,st2)][a][st_]

    def get_P(self):
        P = {}
        S = self.S
        A = self.A
        for state in S:  ##state:(st1, st2)
            Pro1 = self.trans_P(state, 1)  ## transfer probability dict
            Pro2 = self.trans_P(state, 2)  ## trnasfer probability dict
            P[state] = {}
            for a1, a2 in product(A, A):
                P[state][(a1, a2)] = {}
                for st1_ in Pro1[state][a1].keys():
                    for st2_ in Pro2[state][a2].keys():
                        if Pro1[state][a1][st1_] * Pro2[state][a2][st2_] > 0:
                            P[state][(a1, a2)][(st1_, st2_)] = Pro1[state][a1][st1_] * Pro2[state][a2][st2_]
        return P

    def getReward(self, state, action1, action2, reward):
        sumReward = 0
        for state_ in self.P_s1_a_s2[state][(action1, action2)].keys():
            sumReward += reward[state_] * self.P_s1_a_s2[state][(action1,action2)][state_]
        return sumReward

    def calculate(self, state, reward_1):
        rewardMatrix_1 = np.zeros((4,4))
        rewardMatrix_2 = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                rewardMatrix_1[i][j] = self.getReward(state, self.actionDict[i], self.actionDict[j], reward_1)
                # rewardMatrix_2[i][j] = self.getReward(state, self.actionDict[i], self.actionDict[j], reward_2)
        # if state == ((8, 3), (8, 2)):
        #     print(rewardMatrix_1)
        #     input("111")
        rps = nash.Game(rewardMatrix_1)
        eqs =rps.support_enumeration()
        for eq in eqs:
            policyP1_temp = eq[0]
            policyP2_temp = eq[1]
        try:
            utility = rps[policyP1_temp, policyP2_temp]
            utility_1 = utility[0]
            utility_2 = utility[1]
            # if state == ((8, 3), (8, 2)):
            #     print(utility)
            #     print(utility_1)
        except UnboundLocalError:
            print("does not get policy form support_enumeration, will try vertex_enumberation at this time")
            rps = nash.Game(rewardMatrix_1)
            eqs = rps.vertex_enumeration()
            for eq in eqs:
                policyP1_temp = eq[0]
                policyP2_temp = eq[1]
            try:
                utility = rps[policyP1_temp, policyP2_temp]
                utility_1 = utility[0]
                utility_2 = utility[1]
                print("state is: ",state)
                # print("reward is:",reward_1[state], "       ", reward_2[state])
                print("rewardMatrix_1:", rewardMatrix_1)
                # print("rewardMatrix_2:", rewardMatrix_2)
            except UnboundLocalError:
                print("lzzscl")
                input("111")
        return policyP1_temp, policyP2_temp, utility_1, utility_2

    def dict2Vec(self, dict):
        v = []
        for s in self.S:
            v.append(dict[tuple(s)])
        return np.array(v)

    def valueIter(self):
        policyP1_whole = {}
        policyP2_whole = {}
        for decoy in self.decoys:
            reward_1 = self.init_reward_1[decoy]
            # reward_2 = self.init_reward_2[decoy]
            reward_1New = {}
            # reward_2New = {}
            # policyP1_whole[decoy] = {}
            # policyP2_whole[decoy] = {}
            i = 796
            for state in self.S:
                policyP1, policyP2, utility_1, utility_2 = self.calculate(state, reward_1)
                reward_1New[state] = utility_1
                # reward_2New[state] = utility_1
                # filename = "reward" + str(i)+".pkl"
                policyP1_whole[state] = policyP1
                policyP2_whole[state] = policyP2
                # picklefile = open(filename, "wb")
                # pickle.dump(reward_1, picklefile)
                # picklefile.close()
            # while (not(self.checkConverge(reward_1, reward_1New) and self.checkConverge(reward_2, reward_2New))):
            while (not self.checkConverge(reward_1, reward_1New)):
                filename = "reward" + str(i) + ".pkl"
                picklefile = open(filename, "wb")
                pickle.dump(reward_1, picklefile)
                picklefile.close()
                print(i, "th iteration")
                i += 1
                reward_1 = dcp(reward_1New)
                # reward_2 = dcp(reward_2New)
                for state in self.S:
                    policyP1, policyP2, utility_1, utility_2 = self.calculate(state, reward_1)
                    # reward_1New[state] = utility_1
                    reward_1New[state] = utility_1
                    policyP1_whole[state] = policyP1
                    policyP2_whole[state] = policyP2
        return policyP1_whole, policyP2_whole, reward_1New

    def checkConverge(self, V, V_new):
        VVec = self.dict2Vec(V)
        V_newVec = self.dict2Vec(V_new)
        if (np.inner(VVec - V_newVec, VVec - V_newVec) > self.valuethreshold).any():
            return False
        return True

def DictTrans(dict, nashEqu):
    dict_new = {}
    for state in nashEqu.S:
        dict_new[state] = {}
        for i in range(4):
            dict_new[state][nashEqu.actionDict[i]] = dict[state][i]
    return dict_new

if __name__ == '__main__':
    filename = "Set_(9,5).pkl"
    with open(filename, "rb") as f1:
        sureWinningRegion = pickle.load(f1)
    nashEqu = NashEqu(sureWinningRegion, filename='NashEqu_param.json')
    policyP1_whole, policyP2_whole, reward = nashEqu.valueIter()
    # Pi1 = DictTrans(policyP1_whole, nashEqu)
    # Pi2 = DictTrans(policyP2_whole, nashEqu)

    p1Policy = 'p1Policy.pkl'
    p2Policy = 'p2Policy.pkl'
    rewardFinal = 'rewardFinal.pkl'

    picklefile = open(p1Policy, "wb")
    pickle.dump(policyP1_whole, picklefile)
    picklefile.close()

    picklefile = open(p2Policy, "wb")
    pickle.dump(policyP2_whole, picklefile)
    picklefile.close()

    picklefile = open(rewardFinal, "wb")
    pickle.dump(reward, picklefile)
    picklefile.close()