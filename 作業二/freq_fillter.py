import numpy as np
import cv2
from numpy.fft import fft2, ifft2

def frequency_filtering( f, filter, D0, order ):
	nr, nc = f.shape[:2]	

	fp = np.zeros( [ nr, nc ] )				# 前處理
	for x in range( nr ):
		for y in range( nc ):
			fp[x,y] = pow( -1, x + y ) * f[x,y]
	
	F = fft2( fp )							# 離散傅立葉轉換
	G = F.copy( )
	
	if filter == 1:							# 理想低通濾波器
		for u in range( nr ):
			for v in range( nc ):
				dist = np.sqrt( ( u - nr / 2 ) * ( u - nr / 2 ) +
				                ( v - nc / 2 ) * ( v - nc / 2 ) )
				if dist > D0:
					G[u,v] = 0
					
	elif filter == 2:						# 理想高通濾波器
		for u in range( nr ):
			for v in range( nc ):
				dist = np.sqrt( ( u - nr / 2 ) * ( u - nr / 2 ) +
				                ( v - nc / 2 ) * ( v - nc / 2 ) )
				if dist <= D0:
					G[u,v] = 0

	elif filter == 3:						# 高斯低通濾波器
		for u in range( nr ):
			for v in range( nc ):
				dist = np.sqrt( ( u - nr / 2 ) * ( u - nr / 2 ) +
				                ( v - nc / 2 ) * ( v - nc / 2 ) )
				H = np.exp( -( dist * dist ) / ( 2 * D0 * D0 ) )
				G[u,v] *= H	
	
	elif filter == 4:						# 高斯低通濾波器
		for u in range( nr ):
			for v in range( nc ):
				dist = np.sqrt( ( u - nr / 2 ) * ( u - nr / 2 ) +
				                ( v - nc / 2 ) * ( v - nc / 2 ) )
				H = 1 - np.exp( -( dist * dist ) / ( 2 * D0 * D0 ) )
				G[u,v] *= H
					
	elif filter == 5:						# 巴特沃斯低通濾波器
		for u in range( nr ):
			for v in range( nc ):
				dist = np.sqrt( ( u - nr / 2 ) * ( u - nr / 2 ) +
				                ( v - nc / 2 ) * ( v - nc / 2 ) )
				H = 1.0 / ( 1.0 + pow( dist / D0, 2 * order ) )
				G[u,v] *= H
				
	elif filter == 6:						# 巴特沃斯高通濾波器
		for u in range( nr ):
			for v in range( nc ):
				dist = np.sqrt( ( u - nr / 2 ) * ( u - nr / 2 ) +
				                ( v - nc / 2 ) * ( v - nc / 2 ) )
				H = 1.0 - 1.0 / ( 1.0 + pow( dist / D0, 2 * order ) )
				G[u,v] *= H
					
	gp = ifft2( G )							# 反離散傅立葉轉換

	gp2 = np.zeros( [ nr, nc ] )			# 後處理
	for x in range( nr ):
		for y in range( nc ):
			gp2[x,y] = round( pow( -1, x + y ) * np.real( gp[x,y] ), 0 )
	g = np.uint8( np.clip( gp2, 0, 255 ) )

	return g
	
def main( ):
	print( "Filtering in the Frequency Domain" )
	print( "(1) Ideal Lowpass Filter" )
	print( "(2) Ideal Highpass Filter" )
	print( "(3) Gaussian Lowpass Filter" )
	print( "(4) Gaussian Highpass Filter" )
	print( "(5) Butterworth Lowpass Filter" )
	print( "(6) Butterworth Highpass Filter" )
	filter = eval( input( "Please enter your choice: " ) )	
	cutoff = eval( input( "Please enter cutoff frequency: " ) )
	if filter == 5 or filter == 6:
		order = eval( input( "Please enter order: " ) )
	else:
		order = 1
	img1 = cv2.imread("k.png", cv2.IMREAD_GRAYSCALE)
	img2 = frequency_filtering( img1, filter, cutoff, order )
	# cv2.imshow( "Original Image", img1 )
	cv2.imshow( "Filtering in the Frequency Domain", img2 )	
	cv2.waitKey( 0 )

main( )