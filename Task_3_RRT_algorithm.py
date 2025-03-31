import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

class Tree_node():
    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.children = []
        self.parent = None

class RRT_algorithm():
    def __init__(self,start,end,maze,step_size):
        self.root = Tree_node(start[0],start[1])
        self.end = Tree_node(end[0],end[1])
        self.maze = maze
        self.step_size = step_size
        self.nearest_distance = 999999
        self.nearest_node = None
        self.route = []
        self.end_found = False

    def add_node(self, pos_x, pos_y):
        if self.distance(self.end,[pos_x,pos_y]) <= self.step_size:
            self.end.parent = self.nearest_node
            self.nearest_node.children.append(self.end)
            self.end_found = True

        else:
            temp_node = Tree_node(pos_x,pos_y)
            temp_node.parent = self.nearest_node
            self.nearest_node.children.append(temp_node)

    def random_point(self):
        pos_x = random.randint(1,self.maze.shape[1])
        pos_y = random.randint(1,self.maze.shape[0])
        point = np.array([pos_x,pos_y])
        return point

    def required_point(self, start_node, end_point):
        displacement = self.step_size * self.unit_vector(start_node,end_point)
        point = np.array([start_node.pos_x + displacement[0],
                          start_node.pos_y + displacement[1]])
        if self.distance(start_node,end_point) < self.distance(start_node,point):
            return end_point

        if point[0] >= self.maze.shape[1]:
            point[0] = self.maze.shape[1]-1
        if point[1] >= self.maze.shape[0]:
            point[1] = self.maze.shape[0]-1
        return point

    def obstacle_detected(self, start_node, end_point):
        unit_vector = self.unit_vector(start_node,end_point)
        try:
            int_distance = int(self.distance(start_node,end_point))
        except ValueError:
            int_distance = 7
        else:
            int_distance = int(self.distance(start_node,end_point))
        finally:
            for i in range(int_distance):
                temp_x = start_node.pos_x + i * unit_vector[0]
                temp_y = start_node.pos_y + i * unit_vector[1]
                try:
                    val = self.maze[int(temp_y)][int(temp_x)]
                except IndexError:
                    temp_y = self.maze.shape[0]-1
                except ValueError:
                    return True
                else:
                    if val != 255:
                        return True
                finally:
                    pass
            return False

    def unit_vector(self, start_node, end_point):
        distance = self.distance(start_node,end_point)
        return np.array([(end_point[0] - start_node.pos_x)/distance , 
                (end_point[1] - start_node.pos_y)/distance])

    def find_nearest(self, root, point):
        if root == None:
            return 
        
        distance = self.distance(root,point)
        if distance <= self.nearest_distance:
            self.nearest_distance = distance
            self.nearest_node = root
        
        for child in root.children:
            self.find_nearest(child,point)

    def distance(self, node_1, point):
        distance = np.sqrt(((node_1.pos_x - point[0])**2) + ((node_1.pos_y - point[1])**2))
        return distance

    def reset(self):
        self.nearest_distance = 10000
        self.nearest_node = None

    def backtrack(self,node):
        if node.pos_x == self.root.pos_x and node.pos_y == self.root.pos_y:
            self.route.append((int(node.pos_x),int(node.pos_y)))
            return
            
        self.route.append((int(node.pos_x),int(node.pos_y)))
        self.backtrack(node.parent)

maze = cv.imread('task_2/maze.png',0)
_,maze = cv.threshold(maze,250,255,cv.THRESH_BINARY)
maze = maze[16:338,8:448]
solution = maze.copy()
root_point = ([13,318],[152,7])
end_point = ([90,318],[437,287])
plt.figure(figsize=(15,10))
plt.imshow(maze,cmap='gray')

for i in range(2):
    rrt = RRT_algorithm(root_point[i],end_point[i],maze,30)
    for j in range(7000):
        rrt.reset()
        random_point = rrt.random_point()
        rrt.find_nearest(rrt.root,random_point)
        new_point = rrt.required_point(rrt.nearest_node,random_point)
        if not rrt.obstacle_detected(rrt.nearest_node,new_point):
            rrt.add_node(new_point[0],new_point[1])
            plt.plot([rrt.nearest_node.pos_x,new_point[0]],
                    [rrt.nearest_node.pos_y,new_point[1]],'-.')
            if rrt.end_found:
                break
    
    rrt.backtrack(rrt.end)
    print(f"Maze {i+1} SOLUTION FOUND")
    for k in range(len(rrt.route)-2):
        cv.line(solution,rrt.route[k],rrt.route[k+1],(128,0,128),2)
    
cv.imshow('Solution',solution)
plt.show()     
cv.waitKey(0)
cv.destroyAllWindows()