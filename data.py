import enum
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

"""
23  分类
"""
def preprocess_data23():
    #读取KDD-cup网络安全数据,将标签数字化
    df1 = pd.read_csv('../NSL-KDD/KDDTrain+.txt')
    df2 = pd.read_csv('../NSL-KDD/KDDTest+.txt')
    df1.columns = [x for x in range(43)]
    df2.columns = [x for x in range(43)]

    train_cnt = len(df1)

    #将测试集中多余的标签删去（测试集有的攻击类型在训练集中未出现，我们删除这类样本）
    s1 = set(np.array(df1[41]).tolist())
    df2 = df2[df2[41].isin(s1)]
    df = pd.concat([df1,df2])
    #42列无用，删去
    del df[42]
    #获取特征和标签
    labels = df.iloc[:, 41]
    data = df.drop(columns= [41])
    #标签编码
    le = LabelEncoder()
    labels =le.fit_transform(labels).astype(np.int64)
    print(le.classes_)
    #特征编码
    data[1] = le.fit_transform(data[1])
    data[2] = le.fit_transform(data[2])
    data[3] = le.fit_transform(data[3])

    #标签和特征转成numpy数组
    data = np.array(data)
    labels = np.array(labels)

    #特征值归一化
    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    x_train, x_test, y_train,y_test = data[:125972], data[125972:], labels[:125972], labels[125972:]

    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    table_train =  np.concatenate((x_train,  y_train.T), axis = 1)
    table_test =  np.concatenate((x_test,  y_test.T), axis = 1)

    import csv
    with open("nslkdd-train-23.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(table_train)

    with open("nslkdd-test-23.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(table_test)

def preprocess_data():
    #读取KDD-cup网络安全数据,将标签数字化
    columns = (['duration'
        ,'protocol_type'
        ,'service'
        ,'flag'
        ,'src_bytes'
        ,'dst_bytes'
        ,'land'
        ,'wrong_fragment'
        ,'urgent'
        ,'hot'
        ,'num_failed_logins'
        ,'logged_in'
        ,'num_compromised'
        ,'root_shell'
        ,'su_attempted'
        ,'num_root'
        ,'num_file_creations'
        ,'num_shells'
        ,'num_access_files'
        ,'num_outbound_cmds'
        ,'is_host_login'
        ,'is_guest_login'
        ,'count'
        ,'srv_count'
        ,'serror_rate'
        ,'srv_serror_rate'
        ,'rerror_rate'
        ,'srv_rerror_rate'
        ,'same_srv_rate'
        ,'diff_srv_rate'
        ,'srv_diff_host_rate'
        ,'dst_host_count'
        ,'dst_host_srv_count'
        ,'dst_host_same_srv_rate'
        ,'dst_host_diff_srv_rate'
        ,'dst_host_same_src_port_rate'
        ,'dst_host_srv_diff_host_rate'
        ,'dst_host_serror_rate'
        ,'dst_host_srv_serror_rate'
        ,'dst_host_rerror_rate'
        ,'dst_host_srv_rerror_rate'
        ,'attack'
        ,'level'])

    df_train = pd.read_csv('../NSL-KDD/KDDTrain+.txt', header=None)
    df_test = pd.read_csv('../NSL-KDD/KDDTest+.txt', header=None)
    df_train.columns = columns
    df_test.columns = columns

    is_attack_test = df_test.attack.map(lambda a: 0 if a == 'normal' else 1)
    is_attack_train = df_train.attack.map(lambda a: 0 if a == 'normal' else 1)

    df_train['is_attack'] = is_attack_train
    df_test['is_attack'] = is_attack_test

    # lists to hold our attack classifications
    dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
    probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
    privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
    access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']
    # we will use these for plotting below
    attack_labels = ['Normal','DoS','Probe','Privilege','Access']
    # helper function to pass to data frame mapping
    def map_attack(attack):
        if attack in dos_attacks:
            # dos_attacks map to 1
            attack_type = 1
        elif attack in probe_attacks:
            # probe_attacks mapt to 2
            attack_type = 2
        elif attack in privilege_attacks:
            # privilege escalation attacks map to 3
            attack_type = 3
        elif attack in access_attacks:
            # remote access attacks map to 4
            attack_type = 4
        else:
            # normal maps to 0
            attack_type = 0
        return attack_type
    
    attack_map_train = df_train.attack.map(map_attack)
    attack_map_test = df_test.attack.map(map_attack)
    df_train["attack_map"] = attack_map_train
    df_test["attack_map"] = attack_map_test

    # 协议标签转换为数字
    # a = list(set(df_train.protocol_type).union(set(df_test.protocol_type)))
    # a.sort()
    all_protocols = ['icmp', 'tcp', 'udp', ]
    protocol_map = {x : i for (i, x) in enumerate(all_protocols)}
    df_train.protocol_type = df_train.protocol_type.map(protocol_map)
    df_test.protocol_type = df_test.protocol_type.map(protocol_map)
    # 服务标签转换为数字
    # a = list(set(df_train.service).union(set(df_test.service)))
    # a.sort()
    all_services = ['IRC', 'X11', 'Z39_50', 'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois']
    service_map = {x : i for (i, x) in enumerate(all_services)}
    df_train.service = df_train.service.map(service_map)
    df_test.service = df_test.service.map(service_map)
    
    # FLAG标签转换为数字 
    # a = list(set(df_train.flag).union(set(df_test.flag))).sort()
    # a.soert()
    all_flags = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    flag_map = {x : i for (i, x) in enumerate(all_flags)}
    df_train.flag = df_train.flag.map(flag_map)
    df_test.flag = df_test.flag.map(flag_map)

    df_train.to_csv("nslkdd-train-pro.csv", index=False)
    df_test.to_csv("nslkdd-test-pro.csv", index=False)


class NSLKDDDataset(Dataset):
    def __init__(self, df, is_binary):
        # df = pd.read_csv(file_name)
        
        x = df.iloc[:, :41].values
        scaler = Normalizer().fit(x)
        x = scaler.transform(x)

        if is_binary:
            y = df.is_attack.values # 二分类
            self.classes = 2
        else:
            y = df.attack_map.values # 五分类
            self.classes = 5
        
        # y = np.array([4 for i in range(len(y))])

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_datasets(csv_file, is_binary, test_ratio = 0.2):
    df = pd.read_csv(csv_file)
    train, test = train_test_split(df, test_size=0.2)
    # train.to_csv("train-train.csv", index=False)
    # test.to_csv("train-test.csv", index=False)
    return (NSLKDDDataset(train, is_binary), NSLKDDDataset(test, is_binary))