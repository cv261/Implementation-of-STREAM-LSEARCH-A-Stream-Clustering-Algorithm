import numpy as np
import sys
import random
import math

total_data = []
# points: A list of points
# an element in points: points[i]: point[0] - data points, point[1] - centre, point[2] - cost, point[3] weight, point[0] - coordinates
# Assumption : all data poimts have all features
class Clustering(object):
    def __init__(self):
        self.seed = 0
        self.sample1_counter = 0
        self.sample1 = []
        self.SEED = 1
        self.SP = 1
        self.ITER = 3
        self.points = []
        self.data_max = np.array([100, 100, 100000, 3])
        self.data_min = np.array([-5, 0, 1, 2])
        self.data_diff = self.data_max - self.data_min
        self.dim = 4
        self.jjj = 0
        self.max_FL = 1000
        self.centres = []
        self.repr_data = []
    # Returns whether the two data points passed to the function are equal.
    def isIdentical(self, p1, p2):
        if p1 == p2:
            return True
        return False

    def setSeed(self, s1):
        self.seed = s1

    def getData(self):
        self.data = np.genfromtxt(self.file_name, unpack=True, skiprows = 1)

    def getArguments(self):
        self.k1 = sys.argv[1]
        self.k2 = sys.argv[2]
        self.file_name = sys.argv[3]
        self.num = sys.argv[4]
        self.dim = sys.argv[5]
        self.sample_size = sys.argv[6]
        self.out_file_name = sys.argv[7]
        
    #Temperature is in degrees Celsius. Humidity is temperature corrected 
    #relative humidity, ranging from 0-100%. 
    #Light is in Lux (a value of 1 Lux corresponds to moonlight, 400 Lux to a bright office, and 100,000 Lux to full sunlight.) 
    #Voltage is expressed in volts, ranging from 2-3
    def sample_from_stream(self, size, data):
        global total_data
        self.points = []
        for i in range(size):
            sample1 = []
            #index = int(random.uniform(0, len(data)))
            index = i
            #APpending data point
            sample1.append(data[index][-4:])
            sample1[0] = np.divide(np.array(sample1[0]) - self.data_min,self.data_diff)
            total_data.append(np.copy(sample1[0]))
            # Appending centre to 0
            sample1.append(0)
            # Appending cost
            sample1.append(0)
            #Appending weight
            sample1.append(1)
            #Apending coordinate
            sample1.append(index)
            self.points.append(sample1)

    def convert_all_features_numerical(self):
        pass

    def dist(self, p1, p2):
        diff = 0
        dist = np.sum((np.array(p1)-np.array(p2))*(np.array(p1)-np.array(p2)))
        return dist

    def speedy(self, z, k):
        centres = []
        total_cost = z
        min_distance, distance = 0, 0
        centres.append(0)
        self.points[0][1], self.points[0][2] = 0, 0
        k = 1
        for i in range(len(self.points)):
            closest = 0
            min_distance = self.dist(self.points[i][0], self.points[0][0])
            for ii in range(k):
                distance = self.dist(self.points[i][0], self.points[centres[ii]][0])
                if distance < min_distance:
                    closest = ii
                    min_distance = distance
            if random.uniform(0,1) < min_distance * self.points[i][3] / z:
                centres.append(i)
                k += 1
                self.points[i][1], self.points[i][2] = i, 0.0
                total_cost += z
            else:
                self.points[i][1] = centres[closest]
                self.points[i][2] = min_distance * self.points[i][3]
                total_cost += self.points[i][2]
        return total_cost, k

    def speedy2(self, z, k, klim):
        centres = []
        total_cost = z
        min_distance, distance = 0, 0
        centres.append(0)
        self.points[0][1], self.points[0][2] = 0,0
        k = 1
        for i in range(len(self.points)):
            closest = 0
            min_distance = self.dist(self.points[i][0], self.points[0][0])
            for ii in range(k):
                distance = self.dist(self.points[i][0], self.points[centres[ii]][0])
                if distance < min_distance:
                    closest = ii
                    min_distance = distance
            if random.uniform(0,1) < min_distance * self.points[i][3] / z:
                centres.append(i)
                k += 1
                self.points[i][1], self.points[i][2] = i, 0.0
                total_cost += z
            else:
                self.points[i][1] = centres[closest]
                self.points[i][2] = min_distance * self.points[i][3]
                total_cost += self.points[i][2]
            if k > klim:
                return total_cost, k
        return total_cost, k


    def gain(self, x, z, num_centres):
        num_points = len(self.points)
        centres_to_close = 0
        switch_membership = np.zeros(num_points)
        lower = np.zeros(num_points)
        cost_of_opening = 0
        if self.points[x][1] != x:
            cost_of_opening = z

        for i in range(num_points):
            lower[self.points[i][1]] = z
        for i in range(num_points):
            x_cost = self.dist(self.points[i][0], self.points[x][0]) * self.points[i][3]
            current_cost = self.points[i][2] * self.points[i][3]
            if (x_cost < current_cost):
                switch_membership[i] = 1
                cost_of_opening += x_cost - current_cost
            else:
                # we are reducing the savings by closing the current median
                lower[self.points[i][1]] += current_cost - x_cost
         # calculate cost of opening a centre at x
        for i in range(num_points):
            # If the savings are positive, then we reduce the cost and also close the centre
            if lower[i] > 0:
                centres_to_close += 1
                cost_of_opening -= lower[i]

        if cost_of_opening < 0:
            for i in range(num_points):
                if switch_membership[i] or lower[self.points[i][1]] > 0:
                    self.points[i][2] = self.points[i][3] * self.dist(self.points[i][0], self.points[x][0])
                    self.points[i][1] = x
            num_centres = num_centres + 1 - centres_to_close
        else:
            cost_of_opening = 0
        return -1 * cost_of_opening, num_centres

    def FL(self, feasible, numfeasible, z, k, cost, iter, e):
        change = -1 * cost
        self.jjj = 0
        while change/cost < -1 * e and self.jjj <= self.max_FL:
            change = 0.0
            numberOfPoints = len(self.points)
            random.shuffle(feasible)
            for i in range(iter):
                if (iter >= numfeasible):
                    x = int(i % numfeasible)
                else:
                    x = int(random.uniform(0, numfeasible))
                    # x = lrand48() % numfeasible
                delta, k = self.gain(feasible[x], z, k)
                change += delta
            self.jjj += 1
            cost += change
        return cost, k

    # feasible is the arra of centres
    def selectFeasible(self, kmin):
        numFeasible = len(self.points)       
        if numFeasible > self.ITER * kmin * math.log(kmin):
            numFeasible = self.ITER * kmin * math.log(kmin)
        feasible = np.zeros(int(numFeasible)+1, dtype=int)
        if numFeasible == len(self.points):
            feasible[i] = i
        else:
            totalWeight = 0.0
            for i in range(int(numFeasible)):
                totalWeight += self.points[i][3]
            feasibleWeight = []
            for i in range(int(numFeasible)):
                feasibleWeight.append(random.uniform(0,1) * totalWeight)
            feasibleWeight.sort()
            mytotal, ii = 0, 0
            for i in range(len(self.points)):
                mytotal += self.points[i][3]
                while ii < int(numFeasible) and mytotal > feasibleWeight[ii]:
                    feasible[ii] = i
                    ii+= 1
            if ii != int(numFeasible):
                print("Error in feasible centres")
        return int(numFeasible), feasible

    def kmedian(self, kmin, kmax):
        numPoints = len(self.points)
        high_z, low_z = 0,0
        feasible = []
        k = 0
        for i in range(numPoints):
            high_z += self.dist(self.points[i][0], self.points[0][0]) * self.points[i][3]
        z = (low_z + high_z)/2
        # shuffle the points passes to kmedian
        random.shuffle(self.points)
        #k - will get initialized to the number of clusters
        #kmax is the maximum number of clusters
        cost, k = self.speedy2(0.001, k, kmax)
        if k <= kmax:
            return cost, k
        print("Number of clusters after call to speedy2 is " + str(k) + " and z is " + str(z))
        random.shuffle(self.points)
        cost, k = self.speedy(z, k)
        print("Number of clusters after call to speedy is " + str(k))
        # check is there are more centres than points
        if numPoints <= kmax:
            for i in range(numPoints):
                # assign itself as it's centres
                self.points[i][1] = i
                # assign cost as 0
                self.points[i][2] = 2
            return 0, k
        while k < kmin:
            print("Not able to form enough clusters, reduce the cost to open a new centre from z = " + str(z))
            high_z = z
            z = (high_z + low_z)/2
            random.shuffle(self.points)
            cost, k = self.speedy(z, k)
            print("Call to speedy found " + str(k) + " centres")
        numfeasible, feasible = self.selectFeasible(kmin)
        print("Feasible centres selected randomly " + str(feasible))
        #print("feasible centres")
        #print(feasible)
        while True:
            lastcost = cost
            print("Evaluate each feasible data point as a new centre for gains in cost. z = cost of opening a new centres is " + str(z))
            #print("Number of clusters is " + str(k) + " and z is " + str(z))
            cost, k = self.FL(feasible, numfeasible, z, k, cost, int(self.ITER * kmax * math.log(kmax)), 0.1)
            print("Call to FL found " + str(k) + " centres. Number of centres is within safe bound, try a more accurate search")
            if (k <= 1.1 * kmax and k >= 0.9*kmin) or (k <= kmax + 2 and k >= kmin - 2):
                print("k = number of centres is within safe bound, try a more accurate search")
                cost, k = self.FL(feasible, numfeasible, z, k, cost, int(self.ITER * kmax* math.log(kmax)), 0.001)
                print("Call to FL found " + str(k) + " centres. Number of centres is within safe bound, try a more accurate search")
            if k > kmax:
                print("k is greater than kmax, increasing cost to create a new centre.")
                low_z = z
                z = (high_z + low_z)/2
                cost += (z - low_z) * k
            if k < kmin:
                print("k is smaller than kmin, reducing cost to create a new centre.")
                high_z = z
                z = (high_z + low_z)/2
                cost += (z - high_z) * k
            if (k <= kmax and k >= kmin) or high_z - low_z < 0.000001:
                print("Giving up search.")
                return cost, k

    def contcenters(self):
        numPoints = len(self.points)
        cost = 0
        for i in range(numPoints):
            #update coordinates of Ccentre Of Mass of the cluster
            if self.points[i][1] != i:
                relweight = self.points[self.points[i][1]][3] + self.points[i][3]
                relweight = self.points[i][3] / relweight
                for ii in range(self.dim):
                    self.points[self.points[i][1]][0][ii] *= 1 - relweight
                    self.points[self.points[i][1]][0][ii] += self.points[i][0][ii] * relweight
                self.points[self.points[i][1]][3] += self.points[i][3]
        cost = 0
        for i in range(numPoints):
            cost += self.dist(self.points[i][0], self.points[self.points[i][1]][0]) * self.points[i][3]
        return cost

    def print_centres(self):
        centres = []
        for point in self.points:
            centres.append(point[1])

    def get_ssd(self):
        ssd = 0
        for point in self.points:
            ssd += self.dist(point[0], self.points[point[1]][0])
        return ssd

    def maintain_centres(self, data):
        print("Clusters in the chunk: ")
        indices = []
        c = set()
        coordinate = set()
        for point in self.points:
            c.add(tuple(list(self.points[point[1]][0])))
            coordinate.add(self.points[point[1]][4])
            indices.append(point[1])
        c = list(c)
        print(indices)
        print("Centres found in this stream: ")
        for v in c:
            self.centres.append(list(v))
            print(v)
        for coor in list(coordinate):
            self.repr_data.append(data[coor][-4:])
        #print(self.repr_data)

def total_ssq(data, centres):
    n = len(centres)
    ssq = 0
    print("Size of stream: " + str(len(data)) + " data points.")
    for d in data:
        d_ = np.repeat([d], n, 0)
        ssq_ = np.min(dist2(d_,centres),axis=0)
        ssq += ssq_
    return ssq

def dist2(p1,p2):
    diff = p1-p2
    dist = np.sum(np.multiply(diff,diff),1)
    return dist

def run_stream_lsearch():
    global total_data
    cluster = Clustering()
    kmin, kmax = 3, 10
    sample_size = 1000
    num_streams = 20
    max_centres = 1000
    with open("data.txt", "r") as f:
        contents = f.readlines()
    total_data = []
    for j in range(num_streams):
        print("Performing Lsearch on " + str(j) + " chuck of data with size = 1000 data points")
        content = contents[j*1000:(j+1)*1000]
        data = []
        for d in content:
            d = d[0:-1].split(" ")
            try:
                data.append(list(map(float, d[2:])))
            except:
                continue
        cluster.sample_from_stream(sample_size, data)
        cost, k = cluster.kmedian(kmin, kmax)
        cost = cluster.get_ssd()
        print()
        print()
        print("Total ssq measure of points in the chunk is " + str(cost) + " and is k = number of centres is " + str(k))
        cost = cluster.contcenters()
        #print("cost is k is " + str(cost) + str(" ") + str(k))
        cluster.maintain_centres(data)
        if len(cluster.centres) >= max_centres:
            cluster.sample_from_stream(len(cluster.centres), cluster.centres)
            cost, k = cluster.kmedian(kmin, kmax)
            cost = cluster.contcenters()
            cluster.centres = []
            cluster.repr_data = []
            cluster.maintain_centres()
    #print(np.std(np.array(cluster.repr_data), axis=0))
    #print(np.mean(np.array(cluster.repr_data), axis=0))
    #X, Y = np.array(cluster.repr_data)[:,0], np.array(cluster.repr_data)[:,3]
    #corr = np.sum(X-np.mean(X)) * np.sum(Y-np.mean(Y))*1/(20000-1)
    ssq = total_ssq(total_data, cluster.centres)
    print("Total sum of squared distances of each data point in the stream from the closest centre found")
    print("by performing Lsearch on each chunk is " + str(ssq))
    print("Normalized centres are ")
    for c in cluster.centres:
        print(c)
    print()
    print("Centres in unnormalized form acts as the representative data of the stream : ")
    for d in cluster.repr_data:
        print(d)
    print("Mean of the data " + str(np.mean(np.array(cluster.repr_data), axis=0)))
    print("Std_dev of the data is " + str(np.std(np.array(cluster.repr_data), axis=0)))
    return ssq

def main():
    if __name__== "__main__":
        avg = []
        for i in range(1):
            avg.append(run_stream_lsearch())
        print(avg)
        print(np.average(avg))

main()

