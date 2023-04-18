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

<img src="images/fisheye.jpg" alt="Description of image" width="500"/>

Output CubeMap (or Bottom Cube if fisheye image is top-down direction)

<img src="outtests/cubemap.jpg" alt="Description of image" width="1500"/>
<!-- ![panorama.jpg](outtests/panorama.jpg)
![panorama_bottom.jpg](outtests/panorama_bottom.jpg) -->