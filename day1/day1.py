from PIL import Image
import numpy
import os

def problem1():
    img_src = Image.open("mars.png")
    img_src.save("result/mars1.png")

def problem2():
    img_src = Image.new('RGB', (128, 128), (255, 255, 255))
    img_mat = numpy.array(img_src)
    for i in range(img_mat.shape[0]):
        img_mat[i,i] = [0, 0, 0]
    img = Image.fromarray(img_mat)
    img.save("result/p1.png")

def problem3():
    img_src = Image.open("mars.png")
    img_src = img_src.point(lambda i: 255*((float(i)/255)**(1/2.2)))
    img_src.save("result/q3.png")

def test():
    img1 = Image.open('mars.png')
    img2 = Image.open('result/q3.png')
    img1_mat = numpy.array(img1)
    img2_mat = numpy.array(img2)


def day1_run():
    problem1()
    problem2()
    problem3()

if __name__ == '__main__':
    try:
        os.mkdir('result')
    except FileExistsError:
        print('Dir exist!')
    day1_run()
    

