import numpy as np

class Edge_node:
    def __init__(self,idx,parent_idx,y,x,startIdx,is_Adjacent =False):
        self.idx = idx
        self.parent_idx = parent_idx
        self.y = y
        self.x = x
        self.is_Adjacent = is_Adjacent
        self.label = startIdx#해당 엣지가 시작한 요소
        self.endlabel = -1#해당 엣지가 끝난 요소

    def get_parentIdx(self):
        if self.is_Adjacent:
            return False, 0
        else:
            return True, self.parent_idx

    def get_startIdx(self):
        return self.label

    def get_point(self):
        return self.y, self.x
