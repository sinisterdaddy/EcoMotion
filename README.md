# EcoMotion

The objective of this project is to develop an efficient route planning algorithm for electric vehicles (EVs) by employing reinforcement learning techniques. The algorithm takes various factors into consideration, such as battery capacity, motor performance, traffic conditions, and energy consumption, to generate optimal routes that minimize energy usage. 

To implement this algorithm, we leverage the Google Maps API to obtain geolocation data and calculate distances and travel times between different points on the map. It is important to acquire your own Google Maps API key, which can be obtained from the Google Cloud Platform.

The project is implemented in Python and requires several libraries, including TensorFlow, NumPy, pandas, haversine, time, random, math, requests, and urllib. Make sure to install these libraries before running the program.

To get started with the project, download all the necessary Python files: main.py, Environment.py, DoubleDQN.py, battery.py, and motor.py. In the main.py file, you need to specify the start and destination positions either by their names or geocodes. Additionally, you can adjust parameters such as step length and the number of training episodes to suit your requirements.

Before running the program, ensure that you have a stable internet connection and access to the Google Maps API. Please note that excessive querying of the API may result in temporary IP blocking. If you have full access to the API, you can remove the sleep command in the code to expedite the learning process.

The project allows room for creativity and further enhancements. You can model realistic battery and motor systems by incorporating factors such as battery degradation, state of charge (SOC), motor fatigue, and heat conditions. The corresponding Python files, battery.py and motor.py, can be modified to accommodate these features.

You can also explore different neural network architectures or learning algorithms in the DoubleDQN.py and main.py files, respectively. The latter file records training data and saves the learning model and checkpoints in a designated folder.

Furthermore, you have the option to expand the action choices in the step function of Environment.py, allowing for a wider range of navigation options. You can also modify the energy consumption calculation and consider concepts like regenerative braking.

The project utilizes the Double-DQN algorithm for reinforcement learning. The learning agent, represented as an electric vehicle, navigates a grid map by selecting actions such as moving north, east, south, or west. The Q-network and Target-network are utilized to determine the best actions based on the current state. The training process involves exploring the map and gradually transitioning towards exploiting actions with the highest Q-values provided by the Double-DQN model.

The learning environment is designed as a grid map, where each grid represents a specific location on the map. The map is divided into grids to facilitate the discretization required for reinforcement learning. The actual shape of the grid may not be a perfect rectangle due to geographical factors and stride length restrictions.

Interacting with the Google Maps API involves sending requests for directions between different locations. The API returns navigation instructions along with information about duration and distance. The elevation of each location is obtained through the Elevation API. These details are utilized in the energy computation process to estimate the energy consumption between each pair of locations.

The energy consumption is calculated based on the elevation change between two positions, assuming an idealized vehicle model. The reward system aims to minimize energy usage, where reachable steps receive negative rewards proportional to energy consumption.

Once the learning process is complete, the algorithm can be applied to find optimal routes for EVs. The agent, equipped with knowledge gained during training, selects actions that minimize energy consumption, ensuring efficient travel between specified start and destination points.

The potential applications of this project extend to various areas, including transportation planning, logistics, and real-time EV navigation. It contributes to the development of eco-friendly and energy-efficient transportation systems, promoting sustainability and reducing carbon emissions.

In conclusion, this project demonstrates the use of reinforcement learning techniques to optimize electric vehicle route planning. The algorithm, based on the Double-DQN approach, takes into account various factors such as battery capacity, motor performance, traffic conditions, and energy consumption. By utilizing the Google Maps API and incorporating grid-based learning environments, the algorithm provides efficient navigation solutions for EVs. The project offers room for customization and expansion, allowing for the inclusion of additional features and the exploration of different learning algorithms or network architectures.  

References:-</br> 
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0 </br>
https://arxiv.org/abs/1602.02867</br>
https://arxiv.org/pdf/2011.01771.pdf</br>
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
