# PCV-chapter03

记录学习Python Computer Vision的过程

第三次

## Homography（单应性变换）

图像变换的方法很多，单应性变换是其中一种方法，单应性变换会涉及到单应性矩阵。单应性变换的目标是通过给定的几个点（通常是4对点）来得到单应性矩阵。单应性变换是将一个平面内的点映射到另一个平面内的二维投影变换。平面指的是图像或者三维中平面表面，单应性变换的具有很强的实用性，比如图像标配，图像纠正和纹理扭曲，以及创建全景图像。

### 基础变换

- 刚体变换(rigid transformation): 旋转和平移变换/rotation,translation, 3个自由度，点与点之间的距离不变

- 相似变换(similarity transformation): 增加了缩放尺度, 四个自由度，点与点之间的距离比不变。

- 仿射变换(affine transformation): 仿射变换和相似变换近似，不同之处在于相似变换具有单一旋转因子和单一缩放因子，仿射变换具有两个旋转因子和两个缩放因子，因此具有6个自由度. 不具有保角性和保持距离比的性质，但是原图平行线变换后仍然是平行线.

- 投影变换(projective transformation): 也叫作单应性变换。投影变换是齐次坐标下非奇异的线性变换。然而在非齐次坐标系下却是非线性的，这说明齐次坐标的发明是很有价值的。投影变换比仿射变换多2个自由度，具有8个自由度。上面提到的仿射变换具有的“不变”性质，在投影变换中已不复存在了。尽管如此，它还是有一项不变性，那就是在原图中保持共线的3个点，变换后仍旧共线。

- 透视变换: 将3D空间点投影成2D点的变换

### Affine Transformation（仿射变换）

仿射变换就是说一种二维坐标到二维坐标之间的线性变换，然后在变换后还能保持图形的平直性和平行性，通俗的说就是变换后的图形中，直线还是直线，圆弧还是圆弧，而且图形间的相对位置，平行线还有直线的交角都不会改变，一条直线上的几段线段之间的比例关系保持不变。但是这里要提一下，仿射变换不会保持原来的线段长度，和夹角角度不变。

仿射变换可以通过一系列的原子变换的复合来实现，包括：平移（Translation）、缩放（Scale）、翻转（Flip）、旋转（Rotation）和剪切（Shear）。

仿射变换可以用下面公式表示：

![image](https://images2015.cnblogs.com/blog/120296/201602/120296-20160218190320300-1769696826.png)

在上面这个公式中你可以实现平移、缩放、翻转、旋转等变换后的坐标

所以说仿射变换可以理解为经过对坐标轴的放缩，旋转，平移后原坐标在在新坐标域中的值
更简洁的说：仿射变换=线性变换+平移

我们有许多种方法来求这个仿射变换矩阵。下面我们使用对应点来计算仿射变换矩阵，下面是具体函数

```python
def Haffine_from_points(fp,tp):
    """ Find H, affine transformation, such that 
        tp is affine transf of fp. """
    
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
        
    # condition points
    # --from points--
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1]) 
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = dot(C1,fp)
    
    # --to points--
    m = mean(tp[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = dot(C2,tp)
    
    # conditioned points have mean zero, so translation is zero
    A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U,S,V = linalg.svd(A.T)
    
    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]
    
    tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1) 
    H = vstack((tmp2,[0,0,1]))
    
    # decondition
    H = dot(linalg.inv(C2),dot(H,C1))
    
    return H / H[2,2]
```

该函数有有以下参数：

- fp：图像固定点坐标
- tp ：图像变换目标坐标

我们能够使用函数Haffine_from_points()来求出仿射变换,函数Haffine_from_points()会返回给定对应点对的最优仿射变换。

### 图像扭曲

对图像块应用仿射变换，我们将其称为图像扭曲。该操作不仅经常应用在计算机图形学，而且经常出现在计算机视觉算法中。扭曲操作可以使用ＳｃｉＰｙ工具包中的ｎａｄｉｍａｇｅ包来完成。使用这个包进行线性变换Ａ和一个平移量b来对图像块应用仿射变换。演示代码如下：

```python
from scipy import ndimage
from PIL import Image
from pylab import *

im = array(Image.open(r'C:\Users\ZQQ\Desktop\advanced\study\computervision\images\ch02\j03.jpg').convert('L'))
H = array([[1.4,0.05,-100],[0.05,1.5,-100],[0,0,1]])
im2 = ndimage.affine_transform(im,H[:2,:2],(H[0,2],H[1,2]))

figure()
gray()
subplot(121)
axis('off')
imshow(im)
subplot(122)
axis('off')
imshow(im2)
show()
```

原图：

![image](https://github.com/zengqq1997/PCVch03/blob/master/j01.jpg)

结果：

![image](https://github.com/zengqq1997/PCVch03/blob/master/result.jpg)

### Alpha通道

在仿射扭曲的例子中有一个简单的应用是将图像或者图像的一部分放置在另一幅图像中，使得它们能够和指定的区域或者标记物对齐。将扭曲的图像和第二幅图像融合，我们就创建了alpha图像。该图像定义了每个像素从各个图像中获取的像素值成分多少。我们基于以下事实，扭曲的图像是在扭曲区域边界之外以0来填充的图像，来创建一个二值的alpha图像。

在计算机图形学中，一个RGB颜色模型的真彩图形，用由红、绿、蓝三个色彩信息通道合成的，每个通道用了8位色彩深度，共计24位，包含了所有彩色信息。为实现图形的透明效果，采取在图形文件的处理与存储中附加上另一个8位信息的方法，这个附加的代表图形中各个素点透明度的通道信息就被叫做Alpha通道。

Alpha通道使用8位二进制数，就可以表示256级灰度，即256级的透明度。白色（值为255）的Alpha像素用以定义不透明的彩色像素，而黑色（值为0）的Alpha通道像素用以定义透明像素，介于黑白之间的灰度（值为30-255）的Alpha像素用以定义不同程度的半透明像素。因而通过一个32位总线的图形卡来显示带Alpha通道的图形，就可能呈现出透明或半透明的视觉效果。

事实上，我们把需要组合的颜色计算出不含Alpha分量的原始RGB分量然后相加便可。如：两幅图像分别为A和B，由这两幅图像组合而成的图像称为C，则可用如下四元组表示图A和B，三元组表示图像C：

- A：（Ra，Ga，Ba，Alphaa）

- B：（Rb，Gb，Bb，Alphab）

- C：（Rc，Gc，Bc）

根据上述算法，则：

- Rc=Ra*Alphaa+Rb*Alphab

- Gc=Ga*Alphaa+Gb*Alphab

- Bc=Ba*Alphaa+Bb*Alphab

这就是两图像混合后的三原色分量。如果有多幅图像需要混合，则按照以上方法两幅两幅地进行混合。

### 图像中的图像

使用仿射变换将一幅图像放置到另一幅图像中

代码如下：

```python
 # -*- coding: utf-8 -*-
from PCV.geometry import warp, homography
from PIL import  Image
from pylab import *
from scipy import ndimage

# example of affine warp of im1 onto im2

im1 = array(Image.open(r'C:\Users\ZQQ\Desktop\advanced\study\computervision\images\ch03\03.jpg').convert('L'))
im2 = array(Image.open(r'C:\Users\ZQQ\Desktop\advanced\study\computervision\images\ch03\01.jpg').convert('L'))
# set to points
tp = array([[280,574,574,280],[340,340,735,735],[1,1,1,1]])
#tp = array([[675,826,826,677],[55,52,281,277],[1,1,1,1]])
im3 = warp.image_in_image(im1,im2,tp)
figure()
gray()
subplot(141)
axis('off')
imshow(im1)
subplot(142)
axis('off')
imshow(im2)
subplot(143)
axis('off')
imshow(im3)

# set from points to corners of im1
m,n = im1.shape[:2]
fp = array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])
# first triangle
tp2 = tp[:,:3]
fp2 = fp[:,:3]
# compute H
H = homography.Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1,H[:2,:2],
(H[0,2],H[1,2]),im2.shape[:2])
# alpha for triangle
alpha = warp.alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
im3 = (1-alpha)*im2 + alpha*im1_t
# second triangle
tp2 = tp[:,[0,2,3]]
fp2 = fp[:,[0,2,3]]
# compute H
H = homography.Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1,H[:2,:2],
(H[0,2],H[1,2]),im2.shape[:2])
# alpha for triangle
alpha = warp.alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
im4 = (1-alpha)*im3 + alpha*im1_t
subplot(144)
imshow(im4)
axis('off')
show()
```

在其中定义的tp变量是图像变换目标坐标，它按照从左上角逆时针的顺序，定义第一幅图像的四个角的点在第二幅中的位置。

原图：

![image](https://github.com/zengqq1997/PCVch03/blob/master/01.jpg)

![image](https://github.com/zengqq1997/PCVch03/blob/master/03.jpg)

实验结果如下：

![image](https://github.com/zengqq1997/PCVch03/blob/master/placingResult.jpg)

在上面的例子中，对应点对为图像和教室幕布的角点。仿射变换可以将一幅图像进行扭曲，是这对应点对可以完美匹配上。这是因为，仿射变换具有六个自由度。

### 小结

本次实验主要学习了仿射变换，和应用仿射变换

