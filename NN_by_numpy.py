import numpy as np
from struct import unpack
import gzip

#Mnist数据集的读取

def __read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    return img


def __read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab


def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img


def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def load_mnist(x_train_path, y_train_path, x_test_path, y_test_path, normalize=True, one_hot=True):
    image = {
        'train': __read_image(x_train_path),
        'test': __read_image(x_test_path)
    }
    label = {
        'train': __read_label(y_train_path),
        'test': __read_label(y_test_path)
    }
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])
    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])
    return (image['train'], label['train']), (image['test'], label['test'])




class LinearLayer:
    def __init__(self, input_D, output_D):
        self._W = np.random.normal(0, 0.1, (input_D, output_D)) #初始化不能为全0
        self._b = np.random.normal(0, 0.1, (1, output_D))
        self._grad_W = np.zeros((input_D, output_D))
        self._grad_b = np.zeros((1, output_D))

    def forward(self, X):
        return np.matmul(X, self._W) + self._b

    def backward(self, X, grad):
        self._grad_W = np.matmul(X.T, grad)
        self._grad_b = np.matmul(grad.T, np.ones(X.shape[0]))
        return np.matmul(grad, self._W.T)

    def update_L2(self, learn_rate, lamda):
        self._W = self._W - (self._grad_W + self._W * lamda) * learn_rate
        self._b = self._b - self._grad_b * learn_rate

class Relu:
    def __init__(self):
        pass

    def forward(self, X):
        return np.where(X < 0, 0, X)

    def backward(self, X, grad):
        return np.where(X > 0, X, 0) * grad

class Softmax:
    def __init__(self):
        pass

    def forward(self, X):
        exp_X = np.exp(X)
        return (exp_X.T/np.sum(exp_X, axis=1)).T

    def backward(self, X, grad):
        exp_X = np.exp(X)
        temp_X = (exp_X.T/np.sum(exp_X, axis=1)).T
        return (temp_X - temp_X ** 2) * grad

def MSE(y, y_):
    return np.sum((y-y_)**2)/y.size



linear1 = LinearLayer(784, 10)
relu1 = Relu()
linear2 = LinearLayer(10, 10)
softmax2 = Softmax()

x_train_path = 'Mnist/train-images-idx3-ubyte.gz'
y_train_path = 'Mnist/train-labels-idx1-ubyte.gz'
x_test_path = 'Mnist/t10k-images-idx3-ubyte.gz'
y_test_path = 'Mnist/t10k-labels-idx1-ubyte.gz'
(x_train, y_train), (x_test, y_test) = load_mnist(x_train_path, y_train_path, x_test_path, y_test_path)

####超参数设定
batch_size = 1000
epoch = 100
lamda = 0.00001 #正则化强度
learn_rate = 0.1  # 学习率

#训练网络
loss_train = []
loss_test = []
ACC_test = []
for i in range(epoch):
    state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)
    pos = 0
    loss = []
    loss_2 = []
    acc = []
    while pos < x_train.shape[0]:
        x_train_batch = x_train[pos:min(pos+batch_size, x_train.shape[0])]
        y_train_batch = y_train[pos:min(pos+batch_size, y_train.shape[0])]
        o0 = x_train_batch
        a1 = linear1.forward(o0)
        o1 = relu1.forward(a1)
        a2 = linear2.forward(o1)
        o2 = softmax2.forward(a2)
        y = o2
    #获得网络当前输出，计算损失loss
        loss_temp = MSE(y, y_train_batch) + lamda/2*(np.sum(linear1._W**2)+np.sum(linear2._W**2))
        loss.append(loss_temp)

    #反向传播，获取梯度
        grad = (y - y_train_batch)/y.size*2
        grad = softmax2.backward(a2, grad)
        grad = linear2.backward(o1, grad)
        grad = relu1.backward(a1, grad)
        grad = linear1.backward(o0, grad)


     #更新网络中线性层的参数
        linear1.update_L2(learn_rate, lamda)
        linear2.update_L2(learn_rate, lamda)
        pos = pos + batch_size
    #在测试集中验证分类准确率：
        o0 = x_test
        a1 = linear1.forward(o0)
        o1 = relu1.forward(a1)
        a2 = linear2.forward(o1)
        o2 = softmax2.forward(a2)
        y = np.argmax(o2, axis=1)
        y_ = np.argmax(y_test, axis=1)
        loss_temp = MSE(o2, y_test) + lamda / 2 * (np.sum(linear1._W ** 2) + np.sum(linear2._W ** 2))
        count = np.sum((y-y_)==0)
        acc_temp = count/y.size
        acc.append(acc_temp)
        loss_2.append(loss_temp)
    print('Loss for epoch', i+1, ':', np.mean(loss))  # 打印每一个epoch的平均loss
    loss_train.append(np.mean(loss))
    print('Test Loss for epoch', i+1, ':', np.mean(loss_2))  # 打印每一个epoch的平均loss
    loss_test.append(np.mean(loss_2))
    print('ACC for epoch', i+1, ':', np.mean(acc))  # 打印每一个epoch的平均loss
    ACC_test.append(np.mean(acc))

#模型的保存
np.save("dense1_W", linear1._W)
np.save("dense1_b", linear1._b)

np.save("dense2_W", linear2._W)
np.save("dense2_b", linear2._b)

#模型的导入
W1 = np.load("dense1_W.npy")
b1 = np.load("dense1_b.npy")
dense1 = LinearLayer(W1.shape[0], W1.shape[1])
dense1._W = W1
dense1._b = b1

W2 = np.load("dense2_W.npy")
b2 = np.load("dense2_b.npy")
dense2 = LinearLayer(W2.shape[0], W2.shape[1])
dense2._W = W2
dense2._b = b2

#模型的可视化
import matplotlib.pyplot as plt

plt.plot(range(1, epoch+1), loss_train, 'b')
plt.xlabel('epoch')
plt.ylabel('Loss in Train')
plt.savefig('Train_loss.png')
plt.close()

plt.plot(range(1, epoch+1), loss_test, 'r')
plt.xlabel('epoch')
plt.ylabel('Loss in Test')
plt.savefig('Test_loss.png')
plt.close()

plt.plot(range(1, epoch+1), ACC_test, 'g')
plt.xlabel('epoch')
plt.ylabel('Accuracy in Test')
plt.savefig('Test_acc.png')
plt.close()

for i in range(10):
    plt.imshow(linear1._W[:,i].reshape(28,28))
    name = 'W' + str(i) + '.png'
    plt.savefig(name)
    plt.close()

