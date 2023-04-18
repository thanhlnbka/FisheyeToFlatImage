# FisheyeToFlatImage
Fisheye -> Panorama -> Cubemap + Part Panorama

# Requirements

### Install Opencv 

```python
git clone --depth=1 -b 4.7.0 https://github.com/opencv/opencv
cd opencv
mkdir build && cd build
cmake -D CMAKE_INSTALL_PREFIX=/usr ..
make -j$(nproc)
sudo make install
sudo make install
```

# Usage

### Build

```python 
mkdir build && cd build && cmake .. && make
```

### Test 

```python
cd build 
./convert
```

# Results

Input Fisheye
<!-- ![source.jpg](images/source.jpg) -->

<img src="images/source.jpg" alt="Description of image" width="500"/>

Output Part Panorama + Cubemap

<img src="outtests/panorama.jpg" alt="Description of image" width="1500"/>
<img src="outtests/panorama_bottom.jpg" alt="Description of image" width="300"/>
<!-- ![panorama.jpg](outtests/panorama.jpg)
![panorama_bottom.jpg](outtests/panorama_bottom.jpg) -->