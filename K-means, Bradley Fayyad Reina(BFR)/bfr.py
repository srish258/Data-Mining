
import os
import re
import json
import time
import sys
import math
import random
import csv
import itertools
import copy
import multiprocessing
from pyspark import SparkContext
def load_data(input_file_path):
    train_data = []
    for line in open(input_file_path):
        datum = list(map(lambda x : float(x), line.strip('\n').split(",")))
        datum[0] = int(datum[0])
        train_data.append(datum)
    return train_data

def make_predict_dict(K, X, predict_data):
    result = {}
    for i in range(K):
        result[i] = []
    for i in range(len(X)):
        result[predict_data[i]].append(X[i])
    return result

def output_intermediate_result(intermediate, output_file_path):
    titles = ["round_id", "nof_cluster_discard", "nof_point_discard", "nof_cluster_compression", "nof_point_compression", "nof_point_retained"]
    content = []
    for index in range(1, len(intermediate) + 1):
        tmp = []
        for t in titles:
            tmp.append(intermediate[index][t])
        content.append(tmp)
    
    with open(output_file_path,"w") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(titles)
        writer.writerows(content)

def output_cluster_result(DS_set, RS_set, output_file_path):
    middle = []
    data = {}
    for ds_label in DS_set:
        for point_index in DS_set[ds_label].points_index_buffer:
            middle.append((point_index, int(ds_label)))
    for point in RS_set:
        middle.append((point[0], -1))
    middle = sorted(middle, key=lambda x: x[0])
    for pair in middle:
        data[str(pair[0])] = pair[1]
    with open(output_file_path, 'w') as fw:
        json.dump(data,fw)

def inverse_index(X):
    inverseMap = {}
    for i in range(len(X)):
        inverseMap[tuple(X[i])] = i
    return inverseMap

class DiscardSet():
    def __init__(self, label, points):
        self.LABEL = label
        self.D = len(points[0])
        self.N = 0
        self.stale = False
        self.SUM = [0.0] * self.D
        self.SUMSQ = [0.0] * self.D
        self.points_index_buffer = []
        self.merge_points(points)
        self.SUMSQ[0] = 0 # [0] is always 0
        self.SUM[0] = 0   # [0] is always 0
        self.update_statistics()
    
    def merge_points(self, points):
        self.N += len(points)
        for p in points:
            self.points_index_buffer.append(p[0])
            for i in range(1,self.D):
                self.SUM[i] += p[i]
                self.SUMSQ[i] += p[i] ** 2
        self.stale = True
    
    def merge_Cluster(self, cluster):
        self.N += cluster.N
        self.SUM += cluster.SUM
        self.SUMSQ += cluster.SUMSQ
        self.points_index_buffer += cluster.points_index_buffer
        self.stale = True
        
    def update_statistics(self):
        centroid = [0] * self.D
        variance = [0] * self.D
        for i in range(1, self.D):
            centroid[i] = self.SUM[i] / self.N
            variance[i] = math.sqrt((self.SUMSQ[i] / self.N) - (self.SUM[i] / self.N) ** 2)
        self.stale = False
        self.centroid, self.variance = centroid, variance
    
    def mahalanobis_distance(self, point):
        if(self.stale):
            self.update_statistics()
        centroid, variance = self.centroid, self.variance
        distance = 0
        for i in range(1,self.D):
            distance = distance + ((point[i] - centroid[i]) / variance[i]) ** 2 
        distance = math.sqrt(distance)
        return distance


class _KMeans():
    def __init__(self, K = 10, tol=1e-4, max_iter=100, init='random', n_init=5, verbose=False, compulsory=False):
        self.K = K
        self.tol = tol 
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.compulsory = compulsory
    
    def run(self, pid, X):
        kmeans = _KMeans_impl(K=self.K, tol = self.tol, max_iter=self.max_iter, init=self.init, verbose=self.verbose, compulsory=self.compulsory)
        Y = kmeans.fit(X)
        score = kmeans.evaluate_coherence()
        del(kmeans)
        return (Y, score)


    def fit(self, X):
        multiprocessing.freeze_support()
        pool = multiprocessing.Pool()
        cpus = self.n_init
        results = []
        for i in range(0, cpus):
            result = pool.apply_async(self.run, args=(i, X,))
            results.append(result)

        pool.close()
        pool.join()
        resultList = [result.get() for result in results]
        resultList = sorted(resultList, key=lambda x:x[1])
        return resultList[0][0]
        


class _KMeans_impl():
    def __init__(self, K = 10, tol=1e-4, max_iter=100, init='random', verbose=False, compulsory=False):
        self.K = K
        self.tol = tol 
        self.max_iter = max_iter
        self.init = init
        self.verbose = verbose
        self.compulsory = compulsory
        self.centroids = []
        self.centroids_hashmap = {}

    def euclidean_distance(self, a, b):
        a = a[1:]
        b = b[1:]
        c = [pow(i - j,2) for i, j in zip(a, b)]
        return math.sqrt(sum(c))
    
    def initialise_centroid(self, train_data, K):
        centroid = []
        centroid_set = set()
        if self.init == 'random':
            while K > 0:
                index = random.randrange(len(train_data))
                candidate = train_data[index]
                if index in centroid_set:
                    continue 
                K -= 1
                centroid.append(candidate)
                centroid_set.add(index)
        
        else:
            index = random.randrange(len(train_data))
            centroid.append(train_data[index])
            centroid_set.add(train_data[index][0])
            K = K - 1
            while K > 0:
                K = K - 1
                globalMax = 0
                candidate = []
                for point in train_data:
                    if point[0] in centroid_set:
                        continue
                    localMax = 0
                    for center in centroid:
                        localMax = localMax + self.euclidean_distance(point, center)
                    if localMax > globalMax:
                        globalMax = localMax
                        candidate = point
                centroid.append(candidate)
                centroid_set.add(candidate[0])
        self.centroids = centroid
    
    def make_centroid_hashmap(self):
        centroids_hashmap = {}
        for center in self.centroids:
            centroids_hashmap[tuple(center)] = []
        self.centroids_hashmap = centroids_hashmap

    def find_nearest_centroid(self, point):
        min_distance = self.euclidean_distance(point, self.centroids[0])
        min_center = self.centroids[0]
        for i in range(1, len(self.centroids)):
            cur_distance = self.euclidean_distance(point, self.centroids[i])
            if cur_distance < min_distance:
                min_distance = cur_distance
                min_center = self.centroids[i]
        return min_center

    def update_centroid(self):
        new_centroids = []
        for centroid in self.centroids:
            count = len(self.centroids_hashmap[tuple(centroid)])
            if count == 0:
                new_centroids.append(centroid)
                continue
            center = [0] * len(centroid)
            for point in self.centroids_hashmap[tuple(centroid)]:
                center = [ i + j for i,j in zip(center, point)]
            center = [i/count for i in center]
            center[0] = 0
            new_centroids.append(center)
        
        centroid_moving = 0
        for i in range(len(new_centroids)):
            centroid_moving += self.euclidean_distance(new_centroids[i], self.centroids[i])
        self.centroids = new_centroids
        return centroid_moving
    
    def make_predict(self, X):
        index_map = inverse_index(X)
        Y = [0] * len(X)
        label = 0
        for centroid in self.centroids_hashmap:
            for point in self.centroids_hashmap[centroid]:
                Y[index_map[tuple(point)]] = label
            label += 1
        return Y

    def evaluate_coherence(self):
        for centroid in self.centroids_hashmap:
            if self.compulsory and len(self.centroids_hashmap[centroid]) == 0:
                return float('inf')

        non_empty_centroids_hashmap = {}
        for centroid in self.centroids_hashmap:
            if len(self.centroids_hashmap[centroid]) > 0:
                non_empty_centroids_hashmap[centroid] = self.centroids_hashmap[centroid]
        
        S = {}
        for centroid in non_empty_centroids_hashmap:
            temp_sum = 0
            for point in non_empty_centroids_hashmap[centroid]:
                temp_sum += self.euclidean_distance(centroid, point)
            S[centroid] = temp_sum / len(non_empty_centroids_hashmap[centroid])
        
        D = {}
        for centroid in non_empty_centroids_hashmap:
            if centroid not in D:
                D[centroid] = []
            for c in S:
                if c != centroid:
                    D[centroid].append((S[centroid] + S[c])/self.euclidean_distance(centroid, c))
        
        totalSum = 0
        for centroid in D:
            totalSum += max(D[centroid])
        
        return totalSum/len(D)
        


    def fit(self, X):
        K = self.K
        train_data = X
        self.initialise_centroid(train_data, K)
        iter_time = 0
        while iter_time < self.max_iter:
            iter_time = iter_time + 1
            self.make_centroid_hashmap() # reset the hashmap

            # Find the nearest center and add it to the hashmap
            for point in train_data:
                nearest_centroid = self.find_nearest_centroid(point)
                self.centroids_hashmap[tuple(nearest_centroid)].append(point)

            # Update centroids
            centroid_moving = self.update_centroid()
            if self.verbose:
                print("==> Iteration Time: " + str(iter_time))
                print("    Centroid Moving Distance: " + str(centroid_moving))
            if centroid_moving < self.tol:
                break
        Y = self.make_predict(X)
        return Y



def useScikitKmeans(X, K):
    keamns = _KMeans(K=K, tol = 1)
    Y = keamns.fit(X)
    return Y


def displaySet(DS_set):
    return
    print("==> Information of DS Sets")
    for label in DS_set:
        print(DS_set[label].N)
    print("---------------------------")


def load_all_data(paths):
    train_data = []
    for input_file_path in paths:
        train_data = train_data + load_data(input_file_path)
    return train_data


def initialise_DS(X, Y, K):
    middle = {}
    DS_set = {}
    for i in range(K):
        middle[i] = []
    for i in range(len(X)):
        middle[Y[i]].append(X[i])
    for label in middle:
        DS_set[label] = DiscardSet(label, middle[label])
    return DS_set


def find_nearest_DS(DS_set, alpha, point):
    dimension = DS_set[0].D - 1
    THRESHOLD = alpha * math.sqrt(dimension)
    distance = float("inf")
    _label = -1
    for label in DS_set:
        dist = DS_set[label].mahalanobis_distance(point)
        if dist < distance:
            distance = dist
            if dist < THRESHOLD:
                _label = label
    return _label


def find_nearest_CS(CS_set, alpha, point):
    if len(CS_set) == 0:
        return -1
    dimension = CS_set[0].D - 1
    THRESHOLD = alpha * math.sqrt(dimension)
    distance = float("inf")
    _index = -1
    for index in range(len(CS_set)):
        dist = CS_set[index].mahalanobis_distance(point)
        if dist < distance and dist < THRESHOLD:
            distance = dist
            _index = index
    return _index


def sample_data(train_data, sample_ratio, beta):
    _bar = int(len(train_data) * sample_ratio)
    dim = len(train_data[0])
    remains = train_data[_bar:]
    sample_candidate = train_data[:_bar]
    dummy_set = DiscardSet("DUMMY", sample_candidate)
    variance = dummy_set.variance
    centroid = dummy_set.centroid
    sample = []
    possible_outlier = []
    not_select = 0
    for point in sample_candidate:
        is_select = True
        for i in range(1, dim):
            if abs(point[i] - centroid[i]) > beta * variance[i]:
                not_select += 1
                possible_outlier.append(point)
                is_select = False
                break
        if is_select:
            sample.append(point)
    del(dummy_set)
    remains = remains + possible_outlier
    return remains, sample


class BFR:
    def process_one_round(self, train_data, next_line, ROUND_BATCH):
        X = train_data[next_line : next_line + ROUND_BATCH]
        ds_batch_process_add = {}
        cs_batch_process_add = {}

        for point in X:
            nearest_DS_label = find_nearest_DS(self.DS_set, self.alpha, point)
            if nearest_DS_label >= 0: 
                if nearest_DS_label not in ds_batch_process_add:
                    ds_batch_process_add[nearest_DS_label] = []
                ds_batch_process_add[nearest_DS_label].append(point)
            else:
                nearest_CS_index = find_nearest_CS(self.CS_set, self.alpha, point)
                if nearest_CS_index >= 0:
                    if nearest_CS_index not in cs_batch_process_add:
                        cs_batch_process_add[nearest_CS_index] = []
                    cs_batch_process_add[nearest_CS_index].append(point)
                else:
                    self.RS_set.append(point)
        for _label in ds_batch_process_add:
            self.DS_set[_label].merge_points(ds_batch_process_add[_label])
        for _index in cs_batch_process_add:
            self.CS_set[_index].merge_points(cs_batch_process_add[_index])

    def make_CS(self):
        if len(self.RS_set) <= 3 * self.K:
            return
        X = self.RS_set
        keamns = _KMeans(K= 3 * self.K, tol = 1, n_init=1, init="kmeans++")
        Y = keamns.fit(X)
        dict_res = make_predict_dict(K * 3, self.RS_set, Y)
        self.RS_set = []
        for l in dict_res:
            if len(dict_res[l]) == 1:
                self.RS_set.append(dict_res[l][0])
            elif len(dict_res[l]) > 1:
                self.CS_set.append(DiscardSet("CS", dict_res[l]))

    def merge_CS(self):
        alpha = 1 # self.alpha
        if len(self.CS_set) == 0:
            return
        dimension = self.CS_set[0].D
        exist = [True] * len(self.CS_set)
        THRESHOLD = alpha * math.sqrt(dimension) # a small alpha here
        index_combination = list(
            itertools.combinations(list(range(len(self.CS_set))), 2)
        )
        index_combination_distance = []
        cs_batch_process_merge = {}
        for i in range(len(self.CS_set)):
            self.CS_set[i].update_statistics()
        for (s, t) in index_combination:
            s_centroid = self.CS_set[s].centroid
            distance = self.CS_set[t].mahalanobis_distance(s_centroid)
            index_combination_distance.append((s, t, distance))
        index_combination_distance = sorted(
            index_combination_distance, key=lambda x: (x[2], x[0], x[1])
        )

        for (s, t, dis) in index_combination_distance:
            if dis > THRESHOLD:
                break
            if exist[s] == False:
                continue
            if dis < THRESHOLD:
                exist[s] = False
                if t not in cs_batch_process_merge:
                    cs_batch_process_merge[t] = []
                cs_batch_process_merge[t].append(s)

        for i in range(len(self.CS_set)):
            if i not in cs_batch_process_merge:
                continue
            for j in range(len(self.CS_set)):
                if j in cs_batch_process_merge and i in cs_batch_process_merge[j]:
                    cs_batch_process_merge[j] += cs_batch_process_merge[i]
                    del cs_batch_process_merge[i]

        remove_index_list = set()
        for dest in cs_batch_process_merge:
            for source in cs_batch_process_merge[dest]:
                self.CS_set[dest].merge_Cluster(self.CS_set[source])
                remove_index_list.add(source)
        tmp = copy.deepcopy(self.CS_set)
        length = len(self.CS_set)
        del self.CS_set
        self.CS_set = []
        for i in range(length):
            if i not in remove_index_list:
                self.CS_set.append(tmp[i])
        del tmp

    def information_summary(self):
        self.intermediate_info[self.round_no] = {}
        self.intermediate_info[self.round_no]["round_id"] = self.round_no
        self.intermediate_info[self.round_no]["nof_cluster_discard"] = len(self.DS_set)
        self.intermediate_info[self.round_no]["nof_point_discard"] = 0
        for label in self.DS_set:
            self.intermediate_info[self.round_no]["nof_point_discard"] += self.DS_set[
                label
            ].N
        self.intermediate_info[self.round_no]["nof_cluster_compression"] = len(
            self.CS_set
        )
        self.intermediate_info[self.round_no]["nof_point_compression"] = 0
        for index in range(len(self.CS_set)):
            self.intermediate_info[self.round_no][
                "nof_point_compression"
            ] += self.CS_set[index].N
        self.intermediate_info[self.round_no]["nof_point_retained"] = len(self.RS_set)

    def initialisation(self):
        self.round_no += 1
        train_data = load_data(self.files[self.current_file_index])
        self.current_file_index += 1
        train_data, sample = sample_data(train_data, self.sample_ratio, self.beta)
        ROUND_BATCH = int(len(train_data) * 0.1)
        DATA_LENGTH = len(train_data)
        X = sample
        keamns = _KMeans(K=self.K, tol = 1, compulsory=True, n_init=15)
        Y = keamns.fit(X)
        self.DS_set = initialise_DS(X, Y, self.K)
        displaySet(self.DS_set)

        next_line = 0
        while next_line < DATA_LENGTH:
            self.process_one_round(train_data, next_line, ROUND_BATCH)
            next_line += ROUND_BATCH 

        
        self.make_CS()
        self.information_summary()

    def rounds(self):
        if self.current_file_index >= len(self.files):
            return
        self.round_no += 1
        train_data = load_data(self.files[self.current_file_index])
        self.current_file_index += 1
        ROUND_BATCH = int(len(train_data) * 0.1)
        DATA_LENGTH = len(train_data)
        next_line = 0
        while next_line < DATA_LENGTH:
            self.process_one_round(train_data, next_line, ROUND_BATCH)
            next_line += ROUND_BATCH

        self.merge_CS()
        self.make_CS()
        self.information_summary()
        displaySet(self.DS_set)
        self.rounds()  

    def teardown(self):
        ds_batch_process_add = {}
        for cs_index in range(len(self.CS_set)):
            self.CS_set[cs_index].update_statistics()
            centroid_point = self.CS_set[cs_index].centroid
            nearest_label = find_nearest_DS(self.DS_set, float("inf"), centroid_point)
            if nearest_label >= 0:
                if nearest_label not in ds_batch_process_add:
                    ds_batch_process_add[nearest_label] = []
                ds_batch_process_add[nearest_label].append(cs_index)

        
        for ds_label in ds_batch_process_add:
            for cs_index in ds_batch_process_add[ds_label]:
                self.DS_set[ds_label].merge_Cluster(self.CS_set[cs_index])

        output_cluster_result(self.DS_set, self.RS_set, self.cluster_result_filepath)
        output_intermediate_result(
            self.intermediate_info, self.intermediate_filepath
        )

    def run(self):
        self.initialisation()
        self.rounds()
        self.teardown()

    def __init__(self, path, K, sample_ratio, cluster_result_filepath, intermediate_filepath):
        tmp = os.listdir(path)
        file_names = [file_name for file_name in tmp if not file_name.startswith(".")]
        file_names.sort()
        self.files = [os.path.join(path, file_name) for file_name in file_names]
        del tmp
        self.alpha = 4
        self.beta = 3 # beta * sigma
        self.K = K
        self.tol = 1e-4
        self.current_file_index = 0
        self.sample_ratio = sample_ratio
        self.round_no = 0
        self.intermediate_info = {}
        self.cluster_result_filepath = cluster_result_filepath
        self.intermediate_filepath = intermediate_filepath
        self.DS_set = []
        self.RS_set = []
        self.CS_set = []


if __name__ == "__main__":
    time_start = time.time()
    tol = 1e-4
    path = sys.argv[1]
    K = int(sys.argv[2])
    cluster_result_filepath = sys.argv[3]
    intermediate_filepath = sys.argv[4]
    bfr = BFR(path, K, 0.5, cluster_result_filepath, intermediate_filepath)
    bfr.run()
    time_end = time.time()
    print("Duration: ", time_end - time_start, "s")
