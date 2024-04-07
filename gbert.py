import matplotlib.pyplot as plt
import numpy as np

# 给定的九个数组
data = [
    np.random.randint(1, 10, size=5),
    np.random.randint(1, 10, size=5),
    np.random.randint(1, 10, size=5),
    np.random.randint(1, 10, size=5),
    np.random.randint(1, 10, size=5),
    np.random.randint(1, 10, size=5),
    np.random.randint(1, 10, size=5),
    np.random.randint(1, 10, size=5),
    np.random.randint(1, 10, size=5)
]

# 创建一个新的图表
plt.figure(figsize=(10, 6))

# 绘制九个数组的柱状图
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.bar(np.arange(len(data[i])), data[i])
    plt.title("Array {}".format(i+1))

# 调整布局
plt.tight_layout()

# 显示图表
plt.savefig('./fig/test.png')
