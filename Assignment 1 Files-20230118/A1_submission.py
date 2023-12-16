from locale import ABDAY_1
from re import L
import numpy as np
from skimage import io
from skimage.color import rgb2gray 
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
import skimage 
from PIL import Image


def part1_histogram_compute():
    I = io.imread("test.jpg") #读取图片
    I = rgb2gray(I) #变成灰色
    I = img_as_ubyte(I) # 把0-1变成0-255  I是保存图像数据的变量，它正在使用函数转换为无符号字节格式

    h = I.shape[0] #高 shape返回的是一个一维数组 中给 高赋值 为0
    w = I.shape[1] #宽 给宽赋值为1 ；  if 颜色为c 赋值为2 
    hist = np.zeros(64)        # 需要自己算灰度直方图   初始化一个 64 元素的 numpy 零数组来存储直方图值。每个元素对应于直方图的一个 bin
    for i in np.arange(0,h,1):    #  loop through every pixel of the image I (迭代图像中每个像素的嵌套循环)
        for j in np.arange(0,w,1): #  returns an array of integers from 0 up to h-1, incrementing by 1 at each step. 
            num = I[i,j] // (256/64) # 把256位的bin映射到64个bin数组 结果是4 
                                     # divides the gray level of the pixel by the bin size to get the corresponding bin index. 
                                     # 256 is the maximum grayscale value and 64 is the number of bins

            num = int(num)          # 上面num➗出来是浮点数 需要转化成整数  converts the floating-point bin index to an integer.
            hist[num] += 1           # 数一数每个灰度 有多少个 increments the histogram count for the corresponding bin by 1. 
            #This counts the number of pixels that have a gray level within the range covered by that bin.

    hist_np, bins = np.histogram(I,bins=64,range=[0,256])  # 用np算灰度直方图 有库可以算
     # 使用 64 个 bin 和范围 [0, 256]计算图像的直方图 
     # The numpy.histogram() function returns two arrays: hist_np contains the histogram values for each bin, and bins contains the bin edges. 
    # I是计算直方图的输入图像。
     #bins=64指定用于直方图的 bin 数。
     #range=[0,256]指定要包含在直方图中的值的范围。在这种情况下，它包括 0 到 256 之间的所有值，这是 8 位图像中的灰度级范围。



    plt.subplot(1, 2, 1) # 绘制图片 第一个图是1 
    plt.plot(hist) # 创建直方图的线图
    plt.title("plot 1") 

    
    plt.subplot(1, 2, 2) # 绘制图片 第二个图是2 
    plt.plot(hist_np) 
    plt.title("plot 2") 

    plt.show()
    
    pass


def part2_histogram_equalization():

    I = io.imread("test.jpg") 
    I = rgb2gray(I)
    I = img_as_ubyte(I)

    h = I.shape[0]
    w = I.shape[1]

    hist = np.zeros(64)  # 在算灰度直方图
    for i in np.arange(0,h,1):
        for j in np.arange(0,w,1):
            num = I[i,j] // (256/64) 
            num = int(num)
            hist[num] += 1

    H = np.zeros(64)  #计算累加 cumulative （此行初始化一个大小为 64 的数组H，所有元素都初始化为 0。此数组将用于计算输入图像的累积直方图。）
    H[0] = hist[0]    #此行将第一个元素设置H为输入图像直方图第一个元素的值hist
    for n in np.arange(1,64,1): #此循环遍历 的剩余 63 个元素
                                # 第一个参数 ,1指定序列的开始。第二个参数 ,64指定序列的结尾。请注意，此值不包含在序列本身中。
                                #第三个参数 ,1指定序列中数字之间的步长。在这种情况下，由于步长为1，序列将包含 1 到 63 之间的所有整数，包括 1 和 63
        H[n] = H[n-1] + hist[n] #，并通过将 的前一个元素添加到 的当前H元素来计算累积直方图。H更新后每个元素都包含所有先前直方图 bin 以及当前 bin 的总和。
        #                       这会创建一个累积直方图，表示图像中像素强度的累积分布。然后使用累积直方图计算图像的直方图均衡变换。
    
    J = np.zeros((h,w))    # 算均衡之后的图片 此行初始化一个大小为 byJ的数组，所有元素都初始化为 0。该数组将用于存储直方图均衡图像
    for i in np.arange(0,h,1): ##This loop iterates through each row of the image
        for j in np.arange(0,w,1):#This nested loop iterates through each column of the image.
            J[i,j] = np.floor((255.0/(h*w))*H[I[i,j] // 4]+0.5) # 均衡的公式 此行计算图像行i和列处j像素的新像素值
            #将当前像素值除以I[i,j]4，将像素值的范围从 [0, 255] 缩放到 [0, 63] b.将除法结果作为指标放入累积直方图中，H得到累积直方图值。
            #C。缩放累积直方图值以255.0/(h*w)在 0 到 255 之间对其进行归一化，并使用并添加 0.5 将其四舍五入为最接近的整数np.floor。
            #d. 将结果值分配给直方图均衡图像的相应元素J。
            #添加 0.5 可确保正确执行舍入 添加 0.5 可确保如果值的小数部分大于或等于 0.5，则结果值将向上舍入为下一个整数。如果小数部分小于 0.5，则结果值将向下舍入为当前整数。
            # 如果不添加 0.5，舍入操作可能会导致不正确的强度映射，因为某些值在应该向上舍入时可能会向下舍入（反之亦然）。通过添加 0.5，可以正确执行舍入操作，确保每个像素强度都映射到直方图中的正确输出 bin。

    hist_eq = np.zeros(64) # 算均衡之后的直方图 （此行初始化一个hist_eq大小为 64 的数组，所有元素都初始化为 0。此数组将用于计算直方图均衡图像的直方图）
    for i in np.arange(0,h,1): #This loop iterates through each row of the histogram-equalized image.
        for j in np.arange(0,w,1): # This nested loop iterates through each column of the histogram-equalized image.
            num = J[i,j] // 4 #This line divides the current pixel value J[i,j] by 4 to convert the range of pixel values from [0, 255] to [0, 63].
            hist_eq[int(num)] += 1 #hist_eq被初始化为一个 numpy 零数组，长度为 64。该数组将表示均衡图像的直方图。
            #对于均衡图像( equalized image )中的每个像素（即(i,j)图像中的每个坐标），相应的强度值计算为J[i,j]。然后将该强度值除以 4 并使用整数除法（即num = J[i,j] // 4）向下舍入，以便将其映射到输出直方图中的正确 bin。
            #最后，将对应于映射 bin 的元素hist_eq加 1。这是使用代码实现的hist_eq[int(num)] += 1。


    plt.subplots_adjust(hspace=.5) #改变高
    plt.subplots_adjust(wspace=.3) #改变宽 
    plt.subplot(2,2,1)   # 代表第一个图片 
    plt.imshow(I, cmap='gray')  # 第二个参数cmap='gray'指定要用于图像显示的颜色图。在这种情况下，'gray'用于以灰度显示图像。
    plt.title("Cameraman image")

    plt.subplot(2,2,3) #代表第三个图片
    plt.plot(hist)
    plt.title("Cameraman image")


    plt.subplot(2,2,2) # 代表第二个图片
    plt.imshow(J, cmap='gray')
    plt.title("Cameraman after histogram equalization")
 
    plt.subplot(2,2,4) #代表第四个图片 
    plt.plot(hist_eq)
    plt.title("Cameraman after histogram equalization")
    plt.show()


def part3_histogram_comparing():
    """add your code here"""
  
    day = io.imread("day.jpg")
    day = rgb2gray(day) 
    day = img_as_ubyte(day)  
    picture_day, n = np.histogram(day,bins=256,range=(0,256)) 

    h_d = day.shape[0]
    w_d = day.shape[1] #give width the value of 1 


    night = io.imread("night.jpg")  
    night = rgb2gray(night) 
    night = img_as_ubyte(night)
    picture_night, n = np.histogram(night,bins=256,range=(0,256))
    #day是输入图像，假设它是像素值在 [0, 255] 范围内的灰度图像。
     #np.histogram()是计算数组直方图的 NumPy 函数。函数的第一个参数是输入数组（在本例中为day），第二个参数 ( bins) 指定数字bins设置为 256，这意味着直方图将有 256 个 bin，一个对应于输入图像。
    #第三个参数 ( range) 指定应包含在直方图中的输入值的范围。在这种情况下，range设置为 (0, 256)，这意味着 [0, 255] 范围内的所有像素值都将包含在直方图中。
    # 该函数返回两个值：picture_day和n。

    h_n = night.shape[0]
    w_n = night.shape[1]

   
    import math 
    BC=0 #定义BC
    for i in np.arange (256): # 
        BC += math.sqrt((picture_day[i] / (h_d * w_d)) * (picture_night[i] / (h_n * w_n))) 
    print(BC)
        #BC最初设置为零。
#循环使用函数迭代interate从 0 到 255（含）的整数np.arange()。
#对于每个整数i，循环计算两项的乘积：picture_day[i] / (h_d * w_d)和picture_night[i] / (h_n * w_n)。i这些项分别表示图像day和图像中像素值的归一化频率night。然后将这些项相乘以计算i两个图像中像素值的联合概率密度。
# These terms represent the normalized frequency of pixel values i in the day and night images, respectively. The terms are then multiplied together to compute the joint probability density of pixel values i in the two images.
#使用函数计算联合概率密度的平方根math.sqrt()，并将结果添加到BC变量中。
#循环完成后，BC打印的值。
#该变量BC表示 Bhattacharyya 系数，它是两个概率分布之间相似性的度量。在这种情况下，BC用于衡量图像day和night图像的像素值分布之间的相似性。较高的值BC表示两幅图像具有更相似的像素值分布，而较低的值表示它们更不相似。

def part4_histogram_matching():
    """add your code here"""
    day1 = io.imread("day.jpg")
    day1 = rgb2gray(day1) 
    day12 = day1#day12就是第一张图
    day1 = img_as_ubyte(day1) 
    picture_day1, n_day = np.histogram(day1,bins=256,range=(0,256))#两个数组，第二个数组貌似没用
    h_d = day1.shape[0]#480，图片的高
    w_d = day1.shape[1]#640，宽
    #######################################################
    temp_day1 = np.zeros(256) #pi （day的 normalized histogram） 创建一个包含 256 个元素的零数组。这将用于存储累积直方图。
    temp_index = 0 #initializes the index variable to 0.
    temp_1 = 0 #initializes a temporary variable to 0.
    #因为picture_day1是histogram,这个for循环计算的是吧这个array中的每一个元素进行累加，得到cumulative 
    for i in picture_day1:
        temp_day1[temp_index] = temp_1 + i  #adds the current histogram value i to the running sum temp_1, and stores the result in the temp_day1 array at the current index temp_index.

        temp_1 = temp_day1[temp_index] # updates the temporary variable temp_1 to the value just added to temp_day1.
        temp_index = temp_index + 1 # increments the index variable for the next iteration.
    temp_index = 0 #resets the index variable to 0.
    #这个for循环，是得到normalized cumulative histogram.
    for i in temp_day1:#试图算 day normalized cumulative hist 
        temp_day1[temp_index] = i /(h_d * w_d)#算H（i）/MN  divides the current cumulative histogram value i by the total number of pixels in the "day" image (h_d * w_d) to get the normalized cumulative histogram value. The result is stored back in the 
        
    ##################################################################
    night1 = io.imread("night.jpg")  
    night1 = rgb2gray(night1) 
    night12 = night1#night12就是第二张图
    night1 = img_as_ubyte(night1)
    picture_night1, n_night = np.histogram(night1,bins=256,range=(0,256))
    h_n = night1.shape[0]
    w_n = night1.shape[1]
    ###################################################################
    temp_night1 = np.zeros(256) #qi （night的 normalized histogram）
    temp_index = 0
    temp_1 = 0

    for i in picture_night1:
        temp_night1[temp_index] = temp_1 + i

        temp_1 = temp_night1[temp_index]
        temp_index = temp_index + 1
    temp_index = 0
    for i in temp_night1:#试图算 day normalized cumulative hist
        temp_night1[temp_index] = i /(h_n * w_n)#算H（i）/MN
    ############################################################################

    a1 = 0   #This sets the variable a1 to 0, which will be used to keep track of the current index in the normalized cumulative histogram of the reference image 
    A = np.zeros(256)   #创建Alist用来存a1 This creates an array A of 256 zeros, which will be used to store the mapping from pixel values in the input image (day1) to pixel values in the output image.
    index = np.arange(len(temp_night1)) #This creates an array index of the same length as the normalized cumulative histogram of the reference image (night1).
    
    print(len(temp_day1))
    for i in index:
        while temp_day1[i] > temp_night1[a1]:

            a1 = a1 + 1 
            
        A[i] = a1
        #此循环遍历i中的每个索引index，当day1该索引处输入图像 ( ) 的归一化累积直方图的值大于night1当前索引处参考图像 ( )的归一化累积直方图的值时a1，
        # 它递增a1. 一旦循环退出，A[i]被设置为 的最终值a1。这实质上是将输入图像中的像素值映射到输出图像中的i像素值。a1

    out_put = np.zeros((h_n,w_n)) #This creates an array out_put of zeros with the same shape as the output image.

    for I in np.arange(len(day1)):
        for J in np.arange(len(day1[I])):
            temp_a = day1[I,J]        
            out_put[I,J] = A[temp_a]
            #此循环遍历输入图像 ( day1) 中的每个像素，在映射数组中查找相应的像素值A，并将输出图像 ( out_put) 中的相应像素值设置为映射值。
#总的来说，此代码执行直方图匹配以将输入图像中的像素值映射到参考图像中像素值的分布，从而生成与参考图像具有相似直方图的输出图像。


###############################################################
   
    day2 = io.imread("day.jpg")
    day22 = day2#day22就是第4张图
    day2 = img_as_ubyte(day2) 
   

    night2 = io.imread("night.jpg")
    night22 = night2#night22就是第5张图
    night2 = img_as_ubyte(night2) 
    

    out_put1 = io.imread("day.jpg")
    out_put12 = out_put1# out_put12 是第六张图
    out_put1 = np.zeros((h_n,w_n))
    out_put1 = img_as_ubyte(out_put1) 
    out_put1 = np.cumsum(out_put1)
    

    plt.subplot(2,3,1)
    plt.imshow(day12, cmap='gray')
    plt.title("source_gs")

    plt.subplot(2,3,4)
    plt.imshow(day22, cmap='gray')
    plt.title("source_rbg")
    

    plt.subplot(2,3,2)
    plt.imshow(night12, cmap='gray')
    plt.title("template_gs")


    plt.subplot(2,3,5)
    plt.imshow(night22, cmap='gray')
    plt.title("template_rbg")

    plt.subplot(2,3,3)
    plt.title("matched_gs")
    plt.imshow(out_put,cmap='gray')

    plt.subplot(2,3,6)
    plt.title("matched_rgb")
    plt.imshow(out_put12,cmap='gray')     

    plt.show()


'''
    #let PA be temp_day1 
    a1 = 0
    A = np.zeros(256) 
    index = np.arange(len(temp_night1))
    out_put = np.zeros((h_n,w_n))
    
    for I in np.arange(len(day1)):
        for J in np.arange(len(day1[I])):
            temp_a = day1[I,J] 
            a1 = 0
    
            while temp_day1[temp_a] > temp_night1[a1]:
                a1 = a1 + 1 
        
            out_put[I,J] = a1


    a1 = 0
    A = np.zeros(256)   #创建Alist用来存a1
    index = np.arange(len(temp_night1))
    
    print(len(temp_day1))
    for i in index:
        while temp_night1[i] > temp_day1[a1]:

            a1 = a1 + 1 
            
        A[i] = a1

    out_put = np.zeros((h_n,w_n))

    for I in np.arange(len(day1)):
        for J in np.arange(len(day1[I])):
            temp_a = day1[I,J]        
            out_put[I,J] = A[temp_a]
''''' 

    

if __name__ == '__main__':
    part1_histogram_compute()  
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()
   

