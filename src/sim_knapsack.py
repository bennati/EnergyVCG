import numpy as np
from DecisionLogic import BaseDecisionLogic

def knapsack(items, maxweight):
    """
    https://codereview.stackexchange.com/questions/20569/dynamic-programming-solution-to-knapsack-problem
    Solve the knapsack problem by finding the most valuable
    subsequence of `items` that weighs no more than `maxweight`.

    `items` is a sequence of pairs `(value, weight)`, where `value` is
    a number and `weight` is a non-negative integer.

    `maxweight` is a non-negative integer.

    Return a pair whose first element is the sum of values in the most
    valuable subsequence, and whose second element is the subsequence.

    >>> items = [(4, 12), (2, 1), (6, 4), (1, 1), (2, 2)]
    >>> knapsack(items, 15)
    (11, [(2, 1), (6, 4), (1, 1), (2, 2)])
    """
    N = len(items)
    W = maxweight
    # Create an (N+1) by (W+1) 2-d list to contain the running values
    # which are to be filled by the dynamic programming routine.
    # bestvalues[i][j] is the best sum of values for any
    # subsequence of the first i items, whose weights sum
    # to no more than j.
    bestvalues = [[0] * (maxweight + 1)
                  for _ in range(len(items) + 1)]

    # Enumerate through the items and fill in the best-value table
    for i, (agentID,value, weight) in enumerate(items):
        for capacity in range(maxweight + 1):
            # Handle the case where the weight of the current item is greater
            # than the "running capacity" - we can't add it to the knapsack
            if weight > capacity:
                bestvalues[i+1][capacity] = bestvalues[i][capacity]
            else:
                # Otherwise, we must choose between two possible candidate values:
                # 1) the value of "running capacity" as it stands with the last item
                #    that was computed; if this is larger, then we skip the current item
                # 2) the value of the current item plus the value of a previously computed
                #    set of items, constrained by the amount of capacity that would be left
                #    in the knapsack (running capacity - item's weight)
                candidate1 = bestvalues[i][capacity]
                candidate2 = bestvalues[i][capacity - weight] + value

                # Just take the maximum of the two candidates; by doing this, we are
                # in effect "setting in stone" the best value so far for a particular
                # prefix of the items, and for a particular "prefix" of knapsack capacities
                bestvalues[i+1][capacity] = max(candidate1, candidate2)

    # Reconstruction
    # Iterate through the values table, and check
    # to see which of the two candidates were chosen. We can do this by simply
    # checking if the value is the same as the value of the previous row. If so, then
    # we say that the item was not included in the knapsack (this is how we arbitrarily
    # break ties) and simply move the pointer to the previous row. Otherwise, we add
    # the item to the reconstruction list and subtract the item's weight from the
    # remaining capacity of the knapsack. Once we reach row 0, we're done
    reconstruction = []
    j = maxweight
    for i in range(N, 0, -1):
        if bestvalues[i][j] != bestvalues[i - 1][j]:
            reconstruction.append(items[i - 1])
            j -= items[i - 1][2] # subtract capacity

    # Reverse the reconstruction list, so that it is presented
    # in the order that it was given
    reconstruction.reverse()

    # Return the best value, and the reconstruction list
    return bestvalues[len(items)][maxweight], reconstruction

class DecisionLogicSupervisorKnapsack(BaseDecisionLogic):
    """
    Optimize the knapsack problem
    """
    def get_decision(self,perceptions):
        W=max([a["threshold"] for a in perceptions]) # public good threshold
        items=[(a["agentID"],a["cost"],a["value"]) for a in perceptions] # values to maximize and contributions to achieve
        maxcost=max([c for i,c,v in items])
        items=[(i,maxcost-c+1,v) for i,c,v in items] # invert the costs because the knapsack is a maximization problem
        _,vals=knapsack(items,W)
        s=sum([v for i,c,v in vals])
        assert(s<=W)
        excluded=[]
        tmp=vals.copy()
        for i in items: # find users that do not contribute
            if i not in tmp:
                excluded.append(i)
            else:
                tmp.remove(i) # remove the value
        del tmp
        # add an additional contributor in case the threshold is not met
        if(excluded != []       # if not all users contributed
           and s<W):              # and if another contributor is needed
            vals.append(max(excluded, key=lambda x: x[1])) # add the contributor with the lowest cost
        assert(sum([v for i,c,v in vals])>=W)           # successful
        # find what agents contribute
        idx=[i for i,c,v in vals] # take first occurrence, it works even if two agents have the same cost and value
        decisions=[True if p["agentID"] in idx else False for p in perceptions]
        # debug
        assert(sum(decisions)==len(idx))
        assert(sum([p["value"] for p in perceptions if p["agentID"] in idx])>=W)
        self.last_actions=[{"contribution":(a["value"] if d else np.nan),"cost":a["cost"],"agentID":a["agentID"],"contributed":d,"timestep":a["timestep"],"threshold":a["threshold"]} for a,d in zip(perceptions,decisions)]
        return self.last_actions
