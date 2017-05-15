from skimage import io       

fd = open('/home/xi/pic.txt', 'w')
pic = io.imread('/home/xi/pic.jpg')

for c in xrange(3):
    for i in xrange(227):
        for j in xrange(227):
            fd.write(str(pic[i][j][c]) + ' ')
