import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse

def generate_heatmap(data, map_name, title, cmap='coolwarm', resample=False):
    idx_ = np.argsort(data[:, 1])
    data = data[idx_]
    len = data.shape[0]
    data = data[int(0.1 * len):int(0.9 * len)]
    print("data shape: ", data.shape)
    x = data[:, 0]
    y = data[:, 1]

    # Estimate the 2D density of the data
    xy = np.vstack([x, y])
    
    kde = gaussian_kde(xy)
    # sample points from the estimated density
    # z = kde.resample(10000)
    if resample:
        
        # 计算数据范围
        x_min = x.min()
        y_min = y.min()
        x_max = x.max()
        y_max = y.max()

        # 生成二维网格
        x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([x.ravel(), y.ravel()])
        z = kde(positions)
        x = positions[0]
        y = positions[1]
    else:
        z = kde(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    plt.figure(figsize=(8, 8*y_range/x_range))
    plt.scatter(x, y, c=z, s=50, cmap=cmap)
    # 对特定点做星标
    # x_new, y_new = x[4700:4800], y[4700:4800]
    # # # print("test frames")
    # plt.scatter(x_new, y_new, marker='^', s=50, c='k', label='test frames1')
    # x_new, y_new = x[5700:6000], y[5700:6000]
    # plt.scatter(x_new, y_new, marker='v', s=25, c='y', label="test frames2")
    

    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    # plt.show()
    plt.savefig(map_name)
    print("save as ", map_name)

    plt.close()
    plt.figure(figsize=(8, 8*y_range/x_range))
    x_new = [xx for xx in x[::100]]
    y_new = [yy for yy in y[::100]]
    z_new = [i/len(x_new) for i in range(len(x_new))]
    plt.scatter(x_new, y_new, c=z_new, s = 50, cmap='plasma')
    plt.colorbar(label='Timeline')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.savefig('timeline.png')

# # Example data generation
# N = 1000
# data = np.random.randn(N, 2)
# data = np.hstack([data, np.ones((N, 1))])

# # Plot the heatmap
# generate_heatmap(data)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate heatmap for face position')
    parser.add_argument('--csv_file', type=str, default='test_faces_info.csv', help='csv file containing face position data')
    parser.add_argument('--map_name', type=str, default='heatmap.png', help='name of the heatmap file')
    parser.add_argument('--title', type=str, default='Distribution of Teacher Positions', help='title of the heatmap')
    parser.add_argument('--resample', type=bool, default=False, help='resample the data')
    args = parser.parse_args()
    csv_file = args.csv_file
    # data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    # data = data[:, 11]
    # print(data)
    import csv
    data = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            str_data = row['nose_w']
            str_data = str_data[1:-1]
            str_data = str_data.split()
            el = []
            for i in range(len(str_data)):
                str_data[i] = str_data[i].strip('[]')
                if str_data[i] == '':
                    continue
                el.append(float(str_data[i]))
            data.append(np.array(el).astype(float))
    print("data shape: ", np.array(data).shape)
    generate_heatmap(np.array(data), args.map_name, 'Distribution of Teacher Positions', resample=args.resample)
