## Split-Learning复现

参考自：https://github.com/Minki-Kim95/Federated-Learning-and-Split-Learning-with-raspberry-pi

### 目录结构

```bash
├── main.py			# 主文件
├── models.py		# 模型文件
└── split_data.py   # 数据集分割
├── server.py       # 服务器端
├── client.py       # 客户端
```

### main.py

方便起见，将客户端与服务器端模型放在同一个文件中。导致训练模型，变成一个一个客户端的训练。在客户端中不断传递服务器端。

- line 190-257: 第一个循环训练轮数，第二个训练遍历客户端，第三个循环遍历每个客户端的训练集并在测试集尚测试

```bash
# 运行
python3 main.py
```

考虑到实际情况，并非一个一个客户端训练，而是每个客户端上传一个batch_size的输出给服务器端，服务器端再训练返回梯度，每个客户端再训练第二个batch_szie，以此循环。所以这里拆分为了两个部分。先启动服务器端，再启动客户端。

### server.py

由于tensor类型不能序列化操作，方便起见，全部存为本地文件再请求响应

### client.py

- line 172-268: 循环25次batch_size，因为每个客户端的训练集长度不一，所以无法确定定值。
- line 188-191: 当且仅当遍历dataloader至某个位置时，才训练，并且只训练一次就切换为下一个客户端。