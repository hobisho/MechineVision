# git init
echo "# MechineVision" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/hobisho/MechineVision.git
git push -u origin main

# push an existing repository from the command line
git remote add origin https://github.com/hobisho/MechineVision.git
git branch -M main
git push -u origin main

# pip 
pip install -r requirements.txt 


# depth
## 右到左視差圖（調整參數）luoluo
    stereo_right = cv2.StereoSGBM_create(
        minDisparity=0,  # 允許負視差
        numDisparities=16 * 6,
        blockSize=9,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

## 右到左視差圖（調整參數）box
    stereo_left = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 10,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=64 * 2 * 5 ** 2,
    disp12MaxDiff=1,
    speckleWindowSize=200,
    speckleRange=32
)