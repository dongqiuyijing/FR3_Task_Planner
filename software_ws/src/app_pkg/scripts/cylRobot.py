import numpy as np
import math
import tf.transformations as tf
import geometry_msgs.msg
#定义弧度与角度的转换
ToRad = math.pi/180
ToDeg = 180/math.pi

#定义基础方向向量
UX = np.array([1,0,0]).T
UY = np.array([0,1,0]).T
UZ = np.array([0,0,1]).T
n = np.array([0,0,0,0]).T#1x4
o = np.array([0,0,0,0]).T#1x4
a = np.array([0,0,0,0]).T#1x4
p = np.array([0,0,0,0]).T#1x4
R = np.array([n[0:3],o[0:3],a[0:3]]).T#3x3
A = np.array([n,o,a,p]).T#4x4
class myRobot:
	def __init__(self):
		#定义机械臂的DH表
		self.Link = [
	    {'theta':0,'dz':0,   'dx':0,  'alf':0, 	   'az':UZ,'n':n,'o':o,'a':a,'p':p,'R':R,'A':A},
	    {'theta':0,'dz':0,   'dx':140,'alf':math.pi/2, 'az':UZ,'n':n,'o':o,'a':a,'p':p,'R':R,'A':A},
	    {'theta':0,'dz':0,   'dx':114,'alf':0,         'az':UZ,'n':n,'o':o,'a':a,'p':p,'R':R,'A':A},
	    {'theta':0,'dz':-280,'dx':0,  'alf':0,         'az':UZ,'n':n,'o':o,'a':a,'p':p,'R':R,'A':A},
	    {'theta':0,'dz':0,   'dx':-114,'alf':0,         'az':UZ,'n':n,'o':o,'a':a,'p':p,'R':R,'A':A},
	    {'theta':0,'dz':-240,'dx':0,  'alf':0, 	   'az':UZ,'n':n,'o':o,'a':a,'p':p,'R':R,'A':A},
	    {'theta':0,'dz':0,   'dx':102,'alf':math.pi/2, 'az':UZ,'n':n,'o':o,'a':a,'p':p,'R':R,'A':A},
	    {'theta':0,'dz':0,   'dx':102,'alf':-math.pi/2,'az':UZ,'n':n,'o':o,'a':a,'p':p,'R':R,'A':A},
	    {'theta':0,'dz':0,   'dx':100,'alf':0, 	   'az':UZ,'n':n,'o':o,'a':a,'p':p,'R':R,'A':A},
		{'theta':-135,'dz':-81,  'dx':44.05,'alf':0, 	   'az':UZ,'n':n,'o':o,'a':a,'p':p,'R':R,'A':A},
		{'theta':90,'dz':0,   'dx':0,'alf':0, 	   'az':UZ,'n':n,'o':o,'a':a,'p':p,'R':R,'A':A}, #先旋转再平移
		{'theta':0,'dz':0,   'dx':0,'alf':0, 	   'az':UZ,'n':n,'o':o,'a':a,'p':p,'R':R,'A':A} 
	    ]
	    #关节名字           关节角度        链接长度         偏移量        链接扭转角度（Z轴）  主心轴
	#计算每个关节的转化矩阵
	#np.array是定义一个矩阵；。T是矩阵的转制
	def MatrixDhLi(self,i):
		C = np.cos(self.Link[i]['theta']*ToRad)   #输入是弧度制的
		S = np.sin(self.Link[i]['theta']*ToRad)
		Ca = np.cos(self.Link[i]['alf'])
		Sa = np.sin(self.Link[i]['alf'])
		a = self.Link[i]['dz']
		d = self.Link[i]['dx']
		self.Link[i]['n'] = np.array([C,S,0,0])
		self.Link[i]['o'] = np.array([-1*S*Ca,C*Ca,Sa,0])
		self.Link[i]['a'] = np.array([S*Sa,-1*C*Sa,Ca,0])
		self.Link[i]['p'] = np.array([a*C,a*S,d,1]).T
		self.Link[i]['R'] = np.array([self.Link[i]['n'][0:3],self.Link[i]['o'][0:3],self.Link[i]['a'][0:3]]).T
		self.Link[i]['A'] = np.array([self.Link[i]['n'],self.Link[i]['o'],self.Link[i]['a'],self.Link[i]['p']]).T
		#print(np.round(self.Link[i]['n'],2))
		#print(np.round(self.Link[i]['A'],2))
	def forwardKinematics(self,th):#正运动学
		"""参数：六个角度列表，单位：角度"""
		#赋予角度
		self.Link[0]['theta']=th[0]
		self.Link[2]['theta']=th[1]
		self.Link[4]['theta']=th[2]
		self.Link[6]['theta']=th[3]
		self.Link[7]['theta']=th[4]
		self.Link[8]['theta']=th[5]
		self.MatrixDhLi(0)
		for i in range(1,12):
			self.MatrixDhLi(i)
			# print(i,np.round(self.Link[i]['A'],2))
		#计算变换矩阵
		for i in range(1,12):
			self.Link[i]['A']=np.dot(self.Link[i-1]['A'],self.Link[i]['A'])
			self.Link[i]['p']=self.Link[i]['A'][:,3]
			self.Link[i]['n']=self.Link[i]['A'][:,0]
			self.Link[i]['0']=self.Link[i]['A'][:,1]
			self.Link[i]['a']=self.Link[i]['A'][:,2]
			self.Link[i]['R']=self.Link[i]['A'][0:3,0:3]
		print("末端变换矩阵",self.Link[8]['A'])
		# for i in range(9):
		# 	print(self.Link[i]['p'])
	def rotationToAngles(self,i):
    	  #计算出空间坐标
		pos=[0,0,0,0,0,0]
		pos[0]=self.Link[i]['p'][0] # X
		pos[1]=self.Link[i]['p'][1] # Y
		pos[2]=self.Link[i]['p'][2] # Z
		#判断是否为奇异情况
		sy = np.sqrt(self.Link[i]['R'][0,0]**2 + self.Link[i]['R'][1,0]**2)
		if sy > 1e-6 :
    	  	#计算欧拉角
			pos[3]=np.arctan2(self.Link[i]['R'][2,1] , self.Link[i]['R'][2,2])*180/math.pi   # Roll
			pos[4]=np.arctan2(-self.Link[i]['R'][2,0] , sy)   # Pitch
			pos[5]=np.arctan2(self.Link[i]['R'][1,0] , self.Link[i]['R'][0,0])*180/math.pi   # Yaw
		else :
			pos[3]=np.arctan2(-self.Link[i]['R'][1,2] , self.Link[i]['R'][1,1])*180/math.pi   # Roll
			pos[4]=np.arctan2(-self.Link[i]['R'][2,0] , sy)*180/math.pi   # Pitch
			pos[5]=0  # Yaw
		# print("位置是：",np.round(pos,2))
		return pos
	def get_transformation_matrix(self,position, direction):
		direction = direction / np.linalg.norm(direction)
		up = np.array([0, 0, 1])
		if np.all(direction == up):
			up = np.array([0, 1, 0])  # 使用 Y 轴作为替代
		right = np.cross(up, direction)
		right /= np.linalg.norm(right)
		up = np.cross(direction, right)  # 更新上向量
		T = np.array([[right[0],right[1],right[2],position[0]],
				      [up[0],up[1],up[2],position[1]],
					  [direction[0],direction[1],direction[2],position[2]],
                      [0,0,0,1]])
		R = np.array([[right[0],right[1],right[2]],
				      [up[0],up[1],up[2]],
					  [direction[0],direction[1],direction[2]]])
		sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
		if sy > 1e-6 :
    	  	#计算欧拉角
			roll=np.arctan2(R[2,1] , R[2,2])   # Roll
			pitch=np.arctan2(-R[2,0] , sy)   # Pitch
			yaw=np.arctan2(R[1,0] , R[0,0])   # Yaw
		else :
			roll=np.arctan2(-R[1,2] , R[1,1])   # Roll
			pitch=np.arctan2(-R[2,0] , sy)   # Pitch
			yaw=0  # Yaw
		quaternion = tf.quaternion_from_euler(roll,pitch,yaw)
		control_pos=[0,0,0,0,0,0,0]
		control_pos[0] = position[0]  #pos_x
		control_pos[1] = position[1]  #pos_y
		control_pos[2] = position[2]  #pos_z
		control_pos[3] = quaternion[0]  #ori_x
		control_pos[4] = quaternion[1]  #ori_y
		control_pos[5] = quaternion[2]  #ori_z
		control_pos[6] = quaternion[3]  #ori_w
		print(control_pos)
		return control_pos
	#将相机坐标系下的点转换到机械臂上
	def posToMatrix(self,pos,angle):
		"""参数一：位置，单位毫米
		   参数二：欧拉角，角度"""
		yaw = angle[2]*math.pi/180  #Z轴
		pitch = angle[1]*math.pi/180#Y轴
		roll = angle[0]*math.pi/180 #X轴
		Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
				       [np.sin(yaw),np.cos(yaw) ,0],
				       [0           ,0            ,1]])
		Ry = np.array([[np.cos(pitch) ,0,np.sin(pitch)],
				       [0             ,1,0            ],
				       [-np.sin(pitch),0,np.cos(pitch)]])
		Rx = np.array([[1,0          ,0           ],
				       [0,np.cos(roll),-np.sin(roll)],
				       [0,np.sin(roll),np.cos(roll) ]])
		R = np.dot(np.dot(Rz , Ry ) , Rx)
		A = np.eye(4)
		A[:3,:3] = R
		A[:3,3] = np.array([pos[0],pos[1],pos[2]])
		return A
	#异面直线碰撞检测函数
	def segmentDistance(self,seg1Start,seg1End,seg2Start,seg2End):#输入的是向量
		#设置阈值
		thresholdValue = 110 #单位是毫米
		#设置成向量
		p1 = seg1Start[0:3]
		p2 = seg1End[0:3]
		q1 = seg2Start[0:3]
		q2 = seg2End[0:3]
		# print(p1)
		#获取方向向量
		dirVetor1=p2-p1
		dirVetor2=q2-q1
		#截取式测量距离，截取为十份
		for r in range(0,11):
			P1 = p1 + (r/10) * dirVetor1
			for u in range(0,11):
				Q1 = q1 + (u/10) * dirVetor2
				distance = np.linalg.norm(P1 - Q1)
				# print(distance)
				if distance < thresholdValue:
					print(distance)
					print(P1,Q1)
					return True
		return False
	def changeMatrix(self,pose):
		th = 3/4 * math.pi
		A = np.array([[np.cos(th),-np.sin(th),0,0],[np.sin(th),np.cos(th),0,0],[0,0,1,0],[0,0,0,1]])
		B = np.array([[1,0,0,pose[0]],[0,1,0,pose[1]],[0,0,1,pose[2]],[0,0,0,1]])
		C = np.dot(np.linalg.inv(A),B)
		pos = C[0:3,3]
		return pos
	def changeMatrix_camera(self,th,pos):
		th = np.array(th)
		self.forwardKinematics(th)
		A = np.array([[1,0,0,pos[0]-17.5],[0,1,0,pos[1]],[0,0,1,pos[2]],[0,0,0,1]])##这里用的都是毫米作单位
		B = np.dot(self.Link[11]['A'],A)
		pos = B[0:3,3]
		return pos
	def changeMatrix_camera_dir(self,th,dir):
		dir = self.changeMatrix_camera(th,dir) - self.changeMatrix_camera(th,[0,0,0])
		return dir
	def collisionDetection(self,th):
		self.forwardKinematics(th)
		for f in range(0,3):
			for i in range(4,8):
				if self.segmentDistance(self.Link[f]['p'],self.Link[f+1]['p'],self.Link[i]['p'],self.Link[i+1]['p']):
					print("位姿不合理，快停止")
					return True
		if self.segmentDistance(self.Link[4]['p'],self.Link[5]['p'],self.Link[7]['p'],self.Link[8]['p']):
					print("位姿不合理，快停止")
					return True
		print("位姿合理")
		return False
	def point_move(self,pos,dir,th):
		"""输入一个位置和方向向量,外加单独控制的末端角度;输出一个位姿信息Pose"""
		pose_goal = geometry_msgs.msg.Pose()
		if isinstance(dir,list):
			dir = np.array(dir)#改成矩阵
			#定义一个位姿数据，并赋值位置信息
		norm = np.linalg.norm(dir) #求模
		if norm == 0 :
			print("方向向量出错")
			return 0
		dir = dir/norm #单位化
		#沿着Z轴方向投影画圆
		x = np.cos(th*math.pi/180)
		y = np.sin(th*math.pi/180)
		#根据平面方程求解z值
		z = (-dir[0]*x-dir[1]*y)/dir[2]
		#求解旋转矩阵x轴分向量;在圆上取一条半径
		x_axis = np.array([x,y,z])
		x_axis = x_axis /np.linalg.norm(x_axis) #求模单位化
		y_axis = np.cross(dir,x_axis)
		y_axis = y_axis /np.linalg.norm(y_axis) #求模单位化
		#写旋转矩阵，Z轴分向量就是输入的方向向量
		R = np.array([[x_axis[0],y_axis[0],dir[0]],[x_axis[1],y_axis[1],dir[1]],[x_axis[2],y_axis[2],dir[2]]])
		#进行四元数求解
		w = np.sqrt(self.min_num(1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
		x, y, z = 0, 0, 0
		if w > 0:
			x = (R[2, 1] - R[1, 2]) / (4 * w)
			y = (R[0, 2] - R[2, 0]) / (4 * w)
			z = (R[1, 0] - R[0, 1]) / (4 * w)
		else:
			# 计算 w 为负的情况下
			if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
				x = np.sqrt(self.min_num(1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
				y = (R[0, 1] + R[1, 0]) / (4 * x)
				z = (R[0, 2] + R[2, 0]) / (4 * x)
			elif R[1, 1] >= R[2, 2]:
				y = np.sqrt(self.min_num(1 + R[1, 1] - R[0, 0] - R[2, 2])) / 2
				x = (R[0, 1] + R[1, 0]) / (4 * y)
				z = (R[1, 2] + R[2, 1]) / (4 * y)
			else:
				z = np.sqrt(self.min_num(1 + R[2, 2] - R[0, 0] - R[1, 1])) / 2
				x = (R[0, 2] + R[2, 0]) / (4 * z)
				y = (R[1, 2] + R[2, 1]) / (4 * z)
		#单位化
		quater = np.array([x,y,z,w])
		quater = quater / np.linalg.norm(quater) #求模
		pose_goal.orientation.x = quater[0]
		pose_goal.orientation.y = quater[1]
		pose_goal.orientation.z = quater[2]
		pose_goal.orientation.w = quater[3]
		pose_goal.position.x = pos[0]
		pose_goal.position.y = pos[1]
		pose_goal.position.z = pos[2]
		return pose_goal

          	      