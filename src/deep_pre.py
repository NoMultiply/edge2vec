import time
import math
import random
import os
# from multiprocessing import Process
from threading import Thread


class DeepPre:
    in_path: str = ''
    out_path: str = ''
    limit: int = 0
    real_num: int = 0
    friends = []  # List<HashMap<Integer,Integer>>
    friends2 = []  # List<HashMap<Integer,Integer>>
    friends3 = []  # List<HashMap<Integer,Integer>>
    friends4 = []  # List<HashMap<Integer,Integer>>
    friends2matrix = []  # boolean[][]
    all_edge_start = []  # List<Integer>
    all_edge_end = []  # List<Integer>
    edge_start = []  # List<Integer>
    edge_end = []  # List<Integer>
    edge_vector1 = []  # List<String>
    edge_vector2 = []  # List<String>
    thread_num: int = 0
    running_threads: int = 0
    max_size: int = 0
    friend_weight: float = 1
    friend2_weight: float = 0.5
    friend3_weight: float = 0.25
    friend4_weight: float = 0.125
    negative_sample: int = 0
    selected = {}  # HashMap<Integer, Integer>

    def __init__(self, file_in, file_out, limit, max_size, negative_sample):
        self.in_path = file_in
        self.out_path = file_out
        self.limit = limit
        self.max_size = max_size
        self.thread_num = 1  # thread_num
        self.negative_sample = negative_sample

    def read_data(self):
        start_time = time.time()
        with open(self.in_path) as fin:
            lines_g = fin.readlines()
        max_node_index = 0
        for line in lines_g:
            components = line.split()
            start = int(components[0])
            end = int(components[1])

            if start > max_node_index:
                max_node_index = start
            if end > max_node_index:
                max_node_index = end
        self.real_num = self.limit if max_node_index > self.limit else max_node_index
        self.friends = [{} for _ in range(self.real_num + 1)]
        self.friends2matrix = [[False for _ in range(self.real_num + 1)] for _ in range(self.real_num + 1)]
        self.all_edge_start = []
        self.all_edge_end = []

        self.edge_start = []
        self.edge_end = []
        self.edge_vector1 = []
        self.edge_vector2 = []
        for line in lines_g:
            components = line.split()
            start = int(components[0])
            end = int(components[1])
            if start <= self.real_num and end <= self.real_num:
                self.friends[start][end] = 1
                self.all_edge_start.append(start)
                self.all_edge_end.append(end)

                self.friends[end][start] = 1
                self.all_edge_start.append(end)
                self.all_edge_end.append(start)

                self.edge_start.append(start)
                self.edge_end.append(end)
                self.edge_vector1.append('')
                self.edge_vector2.append('')

        end_time = time.time()
        print('reading:', (end_time - start_time) / 1000)
        print('real num:', self.real_num)

    def read_data_link(self):
        start_time = time.time()
        with open(self.in_path + 'limitLink') as fin:
            lines_f = fin.readlines()
        max_node_index = 0
        for line in lines_f:
            components = line.split('\t')
            start = int(components[0])
            end = int(components[1])

            if start > max_node_index:
                max_node_index = start
            if end > max_node_index:
                max_node_index = end

        self.real_num = self.limit if max_node_index > self.limit else max_node_index
        self.friends = [{} for _ in range(self.real_num + 1)]
        self.friends2matrix = [[False for _ in range(self.real_num + 1)] for _ in range(self.real_num + 1)]
        self.all_edge_start = []
        self.all_edge_end = []

        self.edge_start = []
        self.edge_end = []
        self.edge_vector1 = []
        self.edge_vector2 = []
        for line in lines_f:
            components = line.split('\t')
            start = int(components[0])
            end = int(components[1])
            if start <= self.real_num and end <= self.real_num:
                self.friends[start][end] = 1
                self.all_edge_start.append(start)
                self.all_edge_end.append(end)

        end_time = time.time()
        print('reading:', (end_time - start_time) / 1000)
        print('real num:', self.real_num)

    def read_data_sign(self):
        start_time = time.time()
        with open(self.in_path + 'edgelist') as fin:
            lines_f = fin.readlines()
        max_node_index = 0
        for line in lines_f:
            components = line.split('\t')
            start = int(components[0])
            end = int(components[1])

            if start > max_node_index:
                max_node_index = start
            if end > max_node_index:
                max_node_index = end

        self.real_num = self.limit if max_node_index > self.limit else max_node_index
        self.friends = [{} for _ in range(self.real_num + 1)]
        self.friends2matrix = [[False for _ in range(self.real_num + 1)] for _ in range(self.real_num + 1)]
        self.all_edge_start = []
        self.all_edge_end = []

        self.edge_start = []
        self.edge_end = []
        self.edge_vector1 = []
        self.edge_vector2 = []
        for line in lines_f:
            components = line.split('\t')
            start = int(components[0])
            end = int(components[1])
            if start <= self.real_num and end <= self.real_num:
                self.friends[start][end] = 1
                self.all_edge_start.append(start)
                self.all_edge_end.append(end)

        end_time = time.time()
        print('reading:', (end_time - start_time) / 1000)
        print('real num:', self.real_num)

    def calculate(self):
        start_time = time.time()
        threads = []
        step = (self.real_num + 1) // self.thread_num
        for i in range(self.thread_num - 1):
            start = i * step
            end = (i + 1) * step
            threads.append(Friends2Thread(self, start, end, i))
        threads.append(Friends2Thread(self, (self.thread_num - 1) * step, self.real_num + 1, self.thread_num - 1))
        self.running_threads = self.thread_num
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.friends2 = []
        for i in range(self.real_num + 1):
            self.friends2.append({})
        for i in range(self.real_num + 1):
            for j in range(self.real_num + 1):
                if self.friends2matrix[i][j]:
                    self.friends2[i][j] = 1

        for i in range(self.real_num + 1):
            for j in range(self.real_num + 1):
                self.friends2matrix[i][j] = False

        threads = []
        for i in range(self.thread_num - 1):
            start = i * step
            end = (i + 1) * step
            threads.append(Friends3Thread(self, start, end, i))
        threads.append(Friends3Thread(self, (self.thread_num - 1) * step, self.real_num + 1, self.thread_num - 1))
        self.running_threads = self.thread_num
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.friends3 = []
        for i in range(self.real_num + 1):
            self.friends3.append({})
        for i in range(self.real_num + 1):
            for j in range(self.real_num + 1):
                if self.friends2matrix[i][j]:
                    self.friends3[i][j] = 1

        for i in range(self.real_num + 1):
            for j in range(self.real_num + 1):
                self.friends2matrix[i][j] = False
        threads = []
        for i in range(self.thread_num - 1):
            start = i * step
            end = (i + 1) * step
            threads.append(Friends4Thread(self, start, end, i))
        threads.append(Friends4Thread(self, (self.thread_num - 1) * step, self.real_num + 1, self.thread_num - 1))
        self.running_threads = self.thread_num
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.friends4 = []
        for i in range(self.real_num + 1):
            self.friends4.append({})
        for i in range(self.real_num + 1):
            for j in range(self.real_num + 1):
                if self.friends2matrix[i][j]:
                    self.friends4[i][j] = 1

        _sum = []
        for i in range(self.real_num + 1):
            _sum.append(len(self.friends[i]) * 4 + len(self.friends2[i]) * 2 + len(self.friends3[i]))
        copy = _sum.copy()
        copy.sort()
        threshold = copy[self.real_num + 1 - self.max_size]
        self.selected = {}
        count = 0
        for i in range(self.real_num + 1):
            if _sum[i] >= threshold:
                self.selected[i] = count
                count += 1
                if count == self.max_size:
                    break

        end_time = time.time()
        print('calculating:', end_time - start_time)

    def write_data(self):
        start_time = time.time()
        threads = []
        step = len(self.edge_start) // self.thread_num
        for i in range(self.thread_num - 1):
            start = i * step
            end = (i + 1) * step
            threads.append(WriteThread(self, start, end, i))
        threads.append(WriteThread(self, (self.thread_num - 1) * step, len(self.edge_start), self.thread_num - 1))
        self.running_threads = self.thread_num
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        end_time = time.time()
        print('writing:', end_time - start_time)

    def write_single(self):
        writer1 = open(self.out_path + 'trainS.txt', 'w')
        writer2 = open(self.out_path + 'testS.txt', 'w')
        for i in range(len(self.edge_start)):
            if random.random() < 0.95:
                continue
            start_node = self.edge_start[i]
            end_node = self.edge_end[i]
            sb = '1 '
            attributes = {}

            for j in self.friends[start_node]:
                if j in self.selected:
                    attributes[self.selected[j] + 1] = self.friend_weight
            for j in self.friends2[start_node]:
                if j in self.selected:
                    attributes[self.selected[j] + 1] = self.friend2_weight
            for j in self.friends3[start_node]:
                if j in self.selected:
                    attributes[self.selected[j] + 1] = self.friend3_weight
            for j in self.friends[end_node]:
                if j in self.selected:
                    attributes[self.selected[j] + self.max_size + 1] = self.friend_weight
            for j in self.friends2[end_node]:
                if j in self.selected:
                    attributes[self.selected[j] + self.max_size + 1] = self.friend2_weight
            for j in self.friends3[end_node]:
                if j in self.selected:
                    attributes[self.selected[j] + self.max_size + 1] = self.friend3_weight

            attribute_list = list(attributes.items())
            attribute_list.sort(key=lambda x: x[0])
            for k, v in attribute_list:
                sb += str(k) + ':' + str(v) + ' '
            sb += '\n'

            attributes2 = {}
            start_node = self.edge_end[i]
            end_node = self.edge_start[i]
            sb2 = '2 '

            for j in self.friends[start_node]:
                if j in self.selected:
                    attributes2[self.selected[j] + 1] = self.friend_weight
            for j in self.friends2[start_node]:
                if j in self.selected:
                    attributes2[self.selected[j] + 1] = self.friend2_weight
            for j in self.friends3[start_node]:
                if j in self.selected:
                    attributes2[self.selected[j] + 1] = self.friend3_weight
            for j in self.friends[end_node]:
                if j in self.selected:
                    attributes2[self.selected[j] + self.max_size + 1] = self.friend_weight
            for j in self.friends2[end_node]:
                if j in self.selected:
                    attributes2[self.selected[j] + self.max_size + 1] = self.friend2_weight
            for j in self.friends3[end_node]:
                if j in self.selected:
                    attributes2[self.selected[j] + self.max_size + 1] = self.friend3_weight

            attribute_list2 = list(attributes.items())
            attribute_list2.sort(key=lambda x: x[0])
            for k, v in attribute_list2:
                sb2 += str(k) + ':' + str(v) + ' '
            sb2 += '\n'

            if random.random() < 0.8:
                writer = writer1
            else:
                writer = writer2
            writer.write(sb)
            writer.write(sb2)

        writer1.close()
        writer2.close()

    def write_csv(self):
        writer = open(os.path.join(self.out_path, 'data.csv'), 'w')
        writer_train = open(os.path.join(self.out_path, 'train.csv'), 'w')
        writer_test = open(os.path.join(self.out_path, 'test.csv'), 'w')
        writer_negative = open(os.path.join(self.out_path, 'negative.csv'), 'w')
        writer_train_list = open(os.path.join(self.out_path, 'train.txt'), 'w')
        writer_test_list = open(os.path.join(self.out_path, 'test.txt'), 'w')
        writer_negative_list = open(os.path.join(self.out_path, 'negative.txt'), 'w')

        for i in range(len(self.all_edge_start)):
            start = self.all_edge_start[i]
            end = self.all_edge_end[i]
            writer.write(self.csv_string(start, end) + '\n')

        for i in range(len(self.edge_start)):
            start = self.edge_start[i]
            end = self.edge_end[i]

            string1 = self.csv_string(start, end)
            string2 = self.csv_string(end, start)

            if random.random() < 0.2:
                writer_train.write(string1 + '\n')
                writer_train.write(string2 + '\n')
                writer_train_list.write(str(start) + ' ' + str(end) + '\n')
            else:
                writer_test.write(string1 + '\n')
                writer_test.write(string2 + '\n')
                writer_test_list.write(str(start) + ' ' + str(end) + '\n')

        edge_num = len(self.all_edge_start)
        for i in range(self.negative_sample):
            sb = ''
            sb2 = ''
            example = random.randint(0, edge_num - 1)
            start = self.all_edge_start[example]
            end = self.all_edge_end[example]
            sb += self.csv_string(start, end) + ','
            sb2 += str(start) + ',' + str(end) + ' '
            start_friend_num = len(self.friends[start])
            end_friend_num = len(self.friends[end])
            positive = random.randint(0, start_friend_num + end_friend_num - 1)
            if positive < start_friend_num:
                positive_start = start
                positive_end = list(self.friends[start].keys())[positive]
            else:
                positive_end = end
                positive_start = list(self.friends[end].keys())[positive - start_friend_num]

            sb += self.csv_string(positive_start, positive_end) + ','
            sb2 += str(positive_start) + ',' + str(positive_end) + ' '
            negative = 5
            while negative > 0:
                neg = random.randint(0, edge_num - 1)
                neg_start = self.all_edge_start[neg]
                neg_end = self.all_edge_end[neg]

                if neg_start == start or neg_start == end or neg_end == start or neg_end == end:
                    continue
                sb += self.csv_string(neg_start, neg_end)
                sb2 += str(neg_start) + ',' + str(neg_end) + ' '
                if negative > 1:
                    sb += ','
                negative -= 1
            writer_negative.write(sb + '\n')
            writer_negative_list.write(sb2 + '\n')

        writer.close()
        writer_train.close()
        writer_test.close()
        writer_train_list.close()
        writer_test_list.close()
        writer_negative.close()
        writer_negative_list.close()

    def write_csv_train_test(self, new_out, rate):
        writer_train = open(new_out + 'train.csv')
        writer_test = open(new_out + 'test.csv')
        writer_train_list = open(new_out + 'train.txt')
        writer_test_list = open(new_out + 'test.txt')

        for i in range(len(self.edge_start)):
            start = self.edge_start[i]
            end = self.edge_end[i]
            string1 = self.csv_string(start, end)
            string2 = self.csv_string(end, start)

            if random.random() < rate:
                writer_train.write(string1 + '\n')
                writer_train.write(string2 + '\n')
                writer_train_list.write(str(start) + ' ' + str(end) + '\n')
            else:
                writer_test.write(string1 + '\n')
                writer_test.write(string2 + '\n')
                writer_test_list.write(str(start) + ' ' + str(end) + '\n')

        writer_train.close()
        writer_test.close()
        writer_train_list.close()
        writer_test_list.close()

    def write_csv_link_train_test(self, new_out, rate):
        assert new_out is not None and rate is not None
        writer = open(self.out_path + 'data.csv', 'w')
        writer_train = open(self.out_path + 'train-0.20.csv', 'w')
        writer_test = open(self.out_path + 'test-0.20.csv', 'w')
        writer_negative = open(self.out_path + 'negative.csv', 'w')
        writer_negative_list = open(self.out_path + 'negative.txt', 'w')

        with open(self.in_path + '/trainLink-0.2.txt') as fin:
            lines_train = fin.readlines()
        with open(self.in_path + '/testLink-0.2.txt') as fin:
            lines_test = fin.readlines()

        for i in range(len(self.all_edge_start)):
            start = self.all_edge_start[i]
            end = self.all_edge_end[i]
            writer.write(self.csv_string(start, end) + '\n')

        for line in lines_train:
            components = line.split()
            start = int(components[0])
            end = int(components[1])

            string1 = self.csv_string(start, end)
            writer_train.write(string1 + '\n')

        for line in lines_test:
            components = line.split()
            start = int(components[0])
            end = int(components[1])

            string1 = self.csv_string(start, end)
            writer_test.write(string1 + '\n')

        edge_num = len(self.all_edge_start)
        for i in range(self.negative_sample):
            sb = ''
            sb2 = ''
            example = random.randint(0, edge_num - 1)
            start = self.all_edge_start[example]
            end = self.all_edge_end[example]
            sb += self.csv_string(start, end) + ','
            sb2 += str(start) + ',' + str(end) + ' '
            start_friend_num = len(self.friends[start])
            end_friend_num = len(self.friends[end])
            positive = random.randint(0, start_friend_num + end_friend_num - 1)
            if positive < start_friend_num:
                positive_start = start
                positive_end = list(self.friends[start].keys())[positive]
            else:
                positive_end = end
                positive_start = list(self.friends[end].keys())[positive - start_friend_num]

            sb += self.csv_string(positive_start, positive_end) + ','
            sb2 += str(positive_start) + ',' + str(positive_end) + ' '
            negative = 5
            while negative > 0:
                neg = random.randint(0, edge_num - 1)
                neg_start = self.all_edge_start[neg]
                neg_end = self.all_edge_end[neg]

                if neg_start == start or neg_start == end or neg_end == start or neg_end == end:
                    continue
                sb += self.csv_string(neg_start, neg_end)
                sb2 += str(neg_start) + ',' + str(neg_end) + ' '
                if negative > 1:
                    sb += ','
                negative -= 1
            writer_negative.write(sb + '\n')
            writer_negative_list.write(sb2 + '\n')

        writer.close()
        writer_train.close()
        writer_test.close()
        writer_negative.close()
        writer_negative_list.close()

    def write_csv_link_train_test2(self, suffix):
        writer_train = open(self.out_path + 'train-' + suffix + '.csv', 'w')
        writer_test = open(self.out_path + 'test-' + suffix + '.csv', 'w')
        with open(self.in_path + '/trainLink-' + suffix + '.txt') as fin:
            lines_train = fin.readlines()
        with open(self.in_path + '/testLink-' + suffix + '.txt') as fin:
            lines_test = fin.readlines()

        for line in lines_train:
            components = line.split()
            start = int(components[0])
            end = int(components[1])

            string1 = self.csv_string(start, end)
            writer_train.write(string1 + '\n')

        for line in lines_test:
            components = line.split()
            start = int(components[0])
            end = int(components[1])

            string1 = self.csv_string(start, end)
            writer_test.write(string1 + '\n')

        writer_train.close()
        writer_test.close()

    def write_csv_sign(self):
        writer = open(self.out_path + 'data.csv', 'w')
        writer_train = open(self.out_path + 'train-0.20.csv', 'w')
        writer_test = open(self.out_path + 'test-0.20.csv', 'w')
        writer_negative = open(self.out_path + 'negative.csv', 'w')
        writer_negative_list = open(self.out_path + 'negative.txt', 'w')

        with open(self.in_path + '/trainLink-0.2') as fin:
            lines_train = fin.readlines()
        with open(self.in_path + '/testLink-0.2') as fin:
            lines_test = fin.readlines()

        for i in range(len(self.all_edge_start)):
            start = self.all_edge_start[i]
            end = self.all_edge_end[i]
            writer.write(self.csv_string(start, end) + '\n')

        for line in lines_train:
            components = line.split()
            start = int(components[0])
            end = int(components[1])
            if start <= self.limit and end <= self.limit:
                string1 = self.csv_string(start, end)
                writer_train.write(string1 + '\n')

        for line in lines_test:
            components = line.split()
            start = int(components[0])
            end = int(components[1])
            if start <= self.limit and end <= self.limit:
                string1 = self.csv_string(start, end)
                writer_test.write(string1 + '\n')

        edge_num = len(self.all_edge_start)
        for i in range(self.negative_sample):
            sb = ''
            sb2 = ''
            example = random.randint(0, edge_num - 1)
            start = self.all_edge_start[example]
            end = self.all_edge_end[example]
            sb += self.csv_string(start, end) + ','
            sb2 += str(start) + ',' + str(end) + ' '
            start_friend_num = len(self.friends[start])
            end_friend_num = len(self.friends[end])
            positive = random.randint(0, start_friend_num + end_friend_num - 1)
            if positive < start_friend_num:
                positive_start = start
                positive_end = list(self.friends[start].keys())[positive]
            else:
                positive_end = end
                positive_start = list(self.friends[end].keys())[positive - start_friend_num]

            sb += self.csv_string(positive_start, positive_end) + ','
            sb2 += str(positive_start) + ',' + str(positive_end) + ' '
            negative = 5
            while negative > 0:
                neg = random.randint(0, edge_num - 1)
                neg_start = self.all_edge_start[neg]
                neg_end = self.all_edge_end[neg]

                if neg_start == start or neg_start == end or neg_end == start or neg_end == end:
                    continue
                sb += self.csv_string(neg_start, neg_end)
                sb2 += str(neg_start) + ',' + str(neg_end) + ' '
                if negative > 1:
                    sb += ','
                negative -= 1
            writer_negative.write(sb + '\n')
            writer_negative_list.write(sb2 + '\n')

        writer.close()
        writer_train.close()
        writer_test.close()
        writer_negative.close()
        writer_negative_list.close()

    def write_csv_sign2(self, suffix):
        writer_train = open(self.out_path + 'train-' + suffix + '.csv', 'w')
        writer_test = open(self.out_path + 'test-' + suffix + '.csv', 'w')

        with open(self.in_path + '/train-' + suffix) as fin:
            lines_train = fin.readlines()
        with open(self.in_path + '/test-' + suffix) as fin:
            lines_test = fin.readlines()

        for line in lines_train:
            components = line.split('\t')
            start = int(components[0])
            end = int(components[1])
            if start <= self.limit and end <= self.limit:
                string1 = self.csv_string(start, end)
                writer_train.write(string1 + '\n')

        for line in lines_test:
            components = line.split()
            start = int(components[0])
            end = int(components[1])
            if start <= self.limit and end <= self.limit:
                string1 = self.csv_string(start, end)
                writer_test.write(string1 + '\n')

        writer_train.close()
        writer_test.close()

    def writer_csv_pro(self, pro_sample):
        writer_local_start_csv = open(self.out_path + 'localStart.csv', 'w')
        writer_local_end_csv = open(self.out_path + 'localEnd.csv', 'w')
        writer_global_start_csv = open(self.out_path + 'globalStart.csv', 'w')
        writer_global_end_csv = open(self.out_path + 'globalEnd.csv', 'w')

        writer_local = open(self.out_path + 'local.txt', 'w')
        writer_global = open(self.out_path + 'global.txt', 'w')

        edge_num = len(self.all_edge_start)
        for i in range(pro_sample):
            example = random.randint(0, edge_num - 1)
            start = self.all_edge_start[example]
            end = self.all_edge_end[example]
            example_random = random.randint(0, edge_num - 1)
            start_random = self.all_edge_start[example_random]
            end_random = self.all_edge_end[example_random]

            example_random2 = random.randint(0, edge_num - 1)
            start_random2 = len(self.all_edge_start[example_random2])
            end_random2 = len(self.all_edge_end[example_random2])

            cosine = self.cosine_sim(start, end, start_random, end_random)
            cosine2 = self.cosine_sim(start, end, start_random2, end_random2)
            writer_global.write(str(start) + '\t' + str(end) + '\t' + str(start_random) + '\t' + str(end_random)
                                + '\t' + ','.join(map(str, cosine)) + '\n')
            writer_global.write(str(start) + '\t' + str(end) + '\t' + str(start_random2) + '\t' + str(end_random2)
                                + '\t' + ','.join(map(str, cosine2)) + '\n')
            writer_global_start_csv.write(self.csv_string(start, end) + '\n')
            writer_global_start_csv.write(self.csv_string(start, end) + '\n')
            writer_global_end_csv.write(self.csv_string(start_random, end_random) + '\n')
            writer_global_end_csv.write(self.csv_string(start_random2, end_random2) + '\n')

            start_friend_num = len(self.friends[start])
            end_friend_num = len(self.friends[end])

            positive = random.randint(0, start_friend_num + end_friend_num - 1)
            if positive < start_friend_num:
                start_positive = start
                end_positive = list(self.friends[start].keys())[positive]
            else:
                end_positive = end
                start_positive = list(self.friends[end].keys())[positive - start_friend_num]
            writer_local.write(str(start) + '\t' + str(end) + '\t' + str(start_positive)
                               + '\t' + str(end_positive) + '\t' + '1\n')
            writer_local_start_csv.write(self.csv_string(start, end) + '\n')
            writer_local_end_csv.write(self.csv_string(start_positive, end_positive) + '\n')

            while True:
                neg = random.randint(0, edge_num - 1)
                start_negative = self.all_edge_start[neg]
                end_negative = self.all_edge_end[neg]

                if start_negative == start or start_negative == end or end_negative == start or end_negative == end:
                    continue
                writer_local.write(str(start) + '\t' + str(end) + '\t' +
                                   str(start_negative) + '\t' + str(end_negative) + '\t' + '0\n')
                writer_local_start_csv.write(self.csv_string(start, end) + '\n')
                writer_local_end_csv.write(self.csv_string(start_negative, end_negative) + '\n')
                break
        writer_local.close()
        writer_global.close()
        writer_local_start_csv.close()
        writer_local_end_csv.close()
        writer_global_start_csv.close()
        writer_global_end_csv.close()

    def cosine_sim(self, start1, end1, start2, end2):
        attributes1 = {}
        attributes2 = {}
        result = [0.0 for _ in range(6)]
        for j in self.friends[start1]:
            if j in self.selected:
                attributes1[self.selected[j] + 1] = self.friend_weight
        for j in self.friends[end1]:
            if j in self.selected:
                attributes1[self.selected[j] + self.max_size + 1] = self.friend_weight
        for j in self.friends[start2]:
            if j in self.selected:
                attributes2[self.selected[j] + 1] = self.friend_weight
        for j in self.friends[end2]:
            if j in self.selected:
                attributes2[self.selected[j] + self.max_size + 1] = self.friend_weight

        result[0] = self.cosine(attributes1, attributes2)
        result[3] = self.euclidean(attributes1, attributes2)

        for j in self.friends2[start1]:
            if j in self.selected:
                attributes1[self.selected[j] + 1] = self.friend2_weight
        for j in self.friends2[end1]:
            if j in self.selected:
                attributes1[self.selected[j] + self.max_size + 1] = self.friend2_weight
        for j in self.friends2[start2]:
            if j in self.selected:
                attributes2[self.selected[j] + 1] = self.friend2_weight
        for j in self.friends2[end2]:
            if j in self.selected:
                attributes2[self.selected[j] + self.max_size + 1] = self.friend2_weight

        result[1] = self.cosine(attributes1, attributes2)
        result[4] = self.euclidean(attributes1, attributes2)

        for j in self.friends3[start1]:
            if j in self.selected:
                attributes1[self.selected[j] + 1] = self.friend3_weight
        for j in self.friends3[end1]:
            if j in self.selected:
                attributes1[self.selected[j] + self.max_size + 1] = self.friend3_weight
        for j in self.friends3[start2]:
            if j in self.selected:
                attributes2[self.selected[j] + 1] = self.friend3_weight
        for j in self.friends3[end2]:
            if j in self.selected:
                attributes2[self.selected[j] + self.max_size + 1] = self.friend3_weight

        result[2] = self.cosine(attributes1, attributes2)
        result[5] = self.euclidean(attributes1, attributes2)
        return result

    @staticmethod
    def cosine(attributes1, attributes2):
        _sum = 0
        sum1 = 0
        sum2 = 0
        for index in attributes1:
            value1 = attributes1[index]
            value2 = attributes2.get(index)
            if value2:
                _sum += value2 * value1
            sum1 += value1 * value1
        for index in attributes2:
            value2 = attributes2[index]
            sum2 += value2 * value2
        return _sum / math.sqrt(sum1) * math.sqrt(sum2)

    @staticmethod
    def euclidean(attributes1, attributes2):
        _sum = 0
        for index in attributes1:
            value1 = attributes1[index]
            value2 = attributes2.get(index)
            if value2:
                _sum += (value1 - value2) * (value1 - value2)
            else:
                _sum += value1 * value1
        for index in attributes2:
            if index not in attributes1:
                value2 = attributes2[index]
                _sum += value2 * value2
        return math.sqrt(_sum)

    def csv_string(self, start, end):
        start_node = start
        end_node = end
        sb = ''
        attributes = {}
        for j in self.friends[start_node]:
            if j in self.selected:
                attributes[self.selected[j] + 1] = self.friend_weight
        for j in self.friends2[start_node]:
            if j in self.selected:
                attributes[self.selected[j] + 1] = self.friend2_weight
        for j in self.friends3[start_node]:
            if j in self.selected:
                attributes[self.selected[j] + 1] = self.friend3_weight

        for j in self.friends[end_node]:
            if j in self.selected:
                attributes[self.selected[j] + self.max_size + 1] = self.friend_weight
        for j in self.friends2[end_node]:
            if j in self.selected:
                attributes[self.selected[j] + self.max_size + 1] = self.friend2_weight
        for j in self.friends3[end_node]:
            if j in self.selected:
                attributes[self.selected[j] + self.max_size + 1] = self.friend3_weight

        attribute_list = list(attributes.items())
        attribute_list.sort(key=lambda x: x)

        list_index = 0
        if len(attribute_list) < 1:
            index = 2 * self.max_size + 1
        else:
            index = attribute_list[list_index][0]
        for k in range(1, 2 * self.max_size + 1):
            if k < index:
                sb += '0,'
            elif k == index:
                sb += str(attribute_list[list_index][1]) + ','
                list_index += 1
                if list_index < len(attribute_list):
                    index = attribute_list[list_index][0]
                else:
                    index = 2 * self.max_size + 1
        list_index = 0
        if len(attribute_list) < 1:
            index = 2 * self.max_size + 1
        else:
            index = attribute_list[list_index][0]
        for k in range(1, 2 * self.max_size + 1):
            if k < index:
                sb += '1'
            elif k == index:
                sb += '10'
                list_index += 1
                if list_index < len(attribute_list):
                    index = attribute_list[list_index][0]
                else:
                    index = 2 * self.max_size + 1
            if k < 2 * self.max_size:
                sb += ','

        return sb


class Friends2Thread(Thread):

    def __init__(self, pre: DeepPre, start, end, index):
        super(Friends2Thread, self).__init__()
        self.pre = pre
        self._start = start
        self.end = end
        self.index = index

    def run(self):
        for i in range(self._start, self.end):
            for j in self.pre.friends[i]:
                for k in self.pre.friends[j]:
                    if k not in self.pre.friends[i] and i != k:
                        self.pre.friends2matrix[i][k] = True


class Friends3Thread(Thread):

    def __init__(self, pre: DeepPre, start, end, index):
        super(Friends3Thread, self).__init__()
        self.pre = pre
        self._start = start
        self.end = end
        self.index = index

    def run(self):
        for i in range(self._start, self.end):
            for j in self.pre.friends2[i]:
                for k in self.pre.friends[j]:
                    if k not in self.pre.friends[i] and k not in self.pre.friends2[i] and i != k:
                        self.pre.friends2matrix[i][k] = True


class Friends4Thread(Thread):

    def __init__(self, pre: DeepPre, start, end, index):
        super(Friends4Thread, self).__init__()
        self.pre = pre
        self._start = start
        self.end = end
        self.index = index

    def run(self):
        for i in range(self._start, self.end):
            for j in self.pre.friends3[i]:
                for k in self.pre.friends[j]:
                    if k not in self.pre.friends[i] and \
                            k not in self.pre.friends2[i] and \
                            k not in self.pre.friends3[i] and i != k:
                        self.pre.friends2matrix[i][k] = True


class WriteThread(Thread):

    def __init__(self, pre: DeepPre, start, end, index):
        super(WriteThread, self).__init__()
        self.pre = pre
        self._start = start
        self.end = end
        self.index = index

    def run(self):
        edge_start = self.pre.edge_start
        edge_end = self.pre.edge_end
        friends = self.pre.friends
        friends2 = self.pre.friends2
        friends3 = self.pre.friends3
        friend_weight = self.pre.friend_weight
        friend2_weight = self.pre.friend2_weight
        friend3_weight = self.pre.friend3_weight
        out_path = self.pre.out_path
        selected = self.pre.selected
        try:
            writer1 = open(out_path + 'train' + self.index + '.txt', 'w')
            writer2 = open(out_path + 'test' + self.index + '.txt', 'w')

            for i in range(self._start, self.end):
                start_node = edge_start[i]
                end_node = edge_end[i]
                sb = '1|'
                for j in friends[start_node]:
                    if j in selected:
                        sb += str(selected[j]) + ':' + str(friend_weight) + ' '
                for j in friends2[start_node]:
                    if j in selected:
                        sb += str(selected[i] + ':' + str(friend2_weight) + ' ')
                for j in friends3[start_node]:
                    if j in selected:
                        sb += str(selected[i] + ':' + str(friend3_weight) + ' ')
                for j in friends[end_node]:
                    if j in selected:
                        sb += str(selected[j] + self.pre.max_size) + ':' + str(friend_weight) + ' '
                for j in friends2[end_node]:
                    if j in selected:
                        sb += str(selected[j] + self.pre.max_size) + ':' + str(friend2_weight) + ' '
                for j in friends3[end_node]:
                    if j in selected:
                        sb += str(selected[j] + self.pre.max_size) + ':' + str(friend3_weight) + ' '

                start_node = edge_end[i]
                end_node = edge_start[i]
                sb2 = '0|'
                for j in friends[start_node]:
                    if j in selected:
                        sb2 += str(selected[j]) + ':' + str(friend_weight) + ' '
                for j in friends2[start_node]:
                    if j in selected:
                        sb2 += str(selected[j]) + ':' + str(friend2_weight) + ' '
                for j in friends3[start_node]:
                    if j in selected:
                        sb2 += str(selected[j]) + ':' + str(friend3_weight) + ' '
                for j in friends[end_node]:
                    if j in selected:
                        sb2 += str(selected[j] + self.pre.max_size) + ':' + str(friend_weight) + ' '
                for j in friends2[end_node]:
                    if j in selected:
                        sb2 += str(selected[j] + self.pre.max_size) + ':' + str(friend2_weight) + ' '
                for j in friends3[end_node]:
                    if j in selected:
                        sb2 += str(selected[j] + self.pre.max_size) + ':' + str(friend3_weight) + ' '

                if random.random() < 0.2:
                    writer = writer1
                else:
                    writer = writer2
                writer.write(sb + '\n')
                writer.write(sb2 + '\n')
            writer1.close()
            writer2.close()
        except Exception as e:
            print(e)
