# Fully Connected Neural Network
## Requirements
Only Numpy !

## RUN
```bash
git clone https://github.com/MarvinMartin24/Fully-Connected-Neural-Network.git
```
Go to your file directory, and run this command :
```bash
python3 main.py
```
## Usage
```python
net = Network()

net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)

net.fit(x_train, y_train, 1000, 0.1)

out = net.predict(x_train)
print(out)
```
## Implementation : XOR
* Input / Data:


| A | B | XOR |
| --- | --- |--- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

```python3
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
```
* Output / Prediction :
```bash
[[[0.]]

 [[1.]]

 [[1.]]

 [[0.]]]
```
## Understand the Maths
This code is  based on this [Medium](https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65)

## Colaborators
[Omar Aflak](https://github.com/OmarAflak) (Author of the Medium Article)
and Marvin Martin
