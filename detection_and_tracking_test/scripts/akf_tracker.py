import numpy as np
import math

class AugmentKalmanFilter():
    """Augment KF for target tracking.
    """
    
    def __init__(self, time_interval, gate_1, gate_2, max_tracking_times, sigmaax, sigmaay, sigmaox, sigmaoy):
        """The basic configuration parameters are as follow:
            time_interval - Time interval
            sigmaax - Sigma of acceleration noise in the x direction
            sigmaay - Sigma of acceleration noise in the y direction
            noise_q - Acceleration noise matrix
            sigmaox - Sigma of observation noise in the x direction
            sigmaoy - Sigma of observation noise in the y direction
            noise_r - Observation noise matrix
            xtrue - Location of ego vehicle
            gate_associate - Association distance
        """
        
        self.time_interval = time_interval
        
        self.noise_q = np.matrix([[sigmaax ** 2, 0],
                       [0, sigmaay ** 2],])
        self.noise_r = np.matrix([[sigmaox ** 2, 0],
                       [0, sigmaoy ** 2],])
        
        self.xtrue = [0, 0]
        self.gate_associate_person = gate_1
        self.gate_associate_vehicle = gate_2
        self.max_tracking_times = max_tracking_times
        
    def kf_initialize(self, color_map):
        """Initialize akf.
        """
        
        # Initialize state xx and covariance px.
        self.xx = np.matrix([])
        self.px = np.matrix([])
        self.xx_cube = np.matrix([])
        self.xx_class = []
        
        # pre_aug stores the location of targets temporarily.
        self.pre_aug = np.matrix([])
        self.pre_aug_cube = np.matrix([])
        self.pre_aug_class = []
        
        # xx_mistracking stores wrong tracking information, and it gets higher when wrong tracking happens.
        self.xx_mistracking = []
        self.xx_color_idx = []
        self.color_map = color_map
    
    def kf_predict(self):
        """KF predict.
        """
        
        xx = self.xx
        px = self.px
        xx_cube = self.xx_cube
        xx_class = self.xx_class
        
        ti = self.time_interval
        q = self.noise_q
        
        if xx.shape[1] == 0:
            return
        else:
            f = np.matrix([[1, ti, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, ti],
                           [0, 0, 0, 1],])
            g = np.matrix([[0.5 * ti ** 2, 0],
                           [ti, 0],
                           [0, 0.5 * ti ** 2],
                           [0, ti],])
            ff = f
            gg = g
            qq = q
            while ff.shape[0] < xx.shape[0]:
                ler = ff.shape[0]
                lec = ff.shape[1]
                ff = np.row_stack((ff, np.zeros((4, lec))))
                ff = np.column_stack((ff, np.zeros((ler + 4, 4))))
                ii = range(-4, 0)
                jj = range(-4, 0)
                for f_i in range(0, 4):
                    for f_j in range(0, 4):
                        ff[ii[f_i], jj[f_j]] = f[f_i, f_j]
                gg = np.row_stack((gg, np.zeros((4, int(lec / 2)))))
                gg = np.column_stack((gg, np.zeros((ler + 4, 2))))
                ii = range(-4, 0)
                jj = range(-2, 0)
                for g_i in range(0, 4):
                    for g_j in range(0, 2):
                        gg[ii[g_i], jj[g_j]] = g[g_i, g_j]
                qq = np.row_stack((qq, np.zeros((2, int(lec / 2)))))
                qq = np.column_stack((qq, np.zeros((int(ler / 2 + 2), 2))))
                ii = range(-2, 0)
                jj = range(-2, 0)
                for q_i in range(0, 2):
                    for q_j in range(0, 2):
                        qq[ii[q_i], jj[q_j]] = q[q_i, q_j]
            xx = ff * xx
            px = ff * px * ff.T + gg * qq * gg.T
            
            for j in range(0, int(xx_cube.shape[0] / 8)):
                xx_cube[range(8 * j, 8 * j + 8), 0] += xx[4 * j + 1, 0] * ti
                xx_cube[range(8 * j, 8 * j + 8), 2] += xx[4 * j + 3, 0] * ti
            
            self.xx = xx
            self.px = px
            self.xx_cube = xx_cube
            self.xx_class = xx_class
            return
    
    def increase_mistracking(self):
        """Increase xx_mistracking.
        """
        
        xx_mistracking = self.xx_mistracking
        
        if xx_mistracking == []:
            pass
        else:
            for i in range(len(xx_mistracking)):
                xx_mistracking[i] += 1
        
        self.xx_mistracking = xx_mistracking
        return
    
    def associate(self, z, z_cube, z_class):
        """Associate.
            Initialize za, id_za and zu.
            za represents targets which have been observed and associated.
            id_za stores indexes of targets in xx.
            zu represents targets which have been observed but not associated.
        """
        
        self.za = np.matrix([])
        self.za_cube = np.matrix([])
        self.za_class = []
        self.id_za = []
        
        self.zu = np.matrix([])
        self.zu_cube = np.matrix([])
        self.zu_class = []
        
        za = self.za
        za_cube = self.za_cube
        za_class = self.za_class
        id_za = self.id_za
        
        zu = self.zu
        zu_cube = self.zu_cube
        zu_class = self.zu_class
        
        xx = self.xx
        xx_mistracking = self.xx_mistracking
        gate_associate_person = self.gate_associate_person
        gate_associate_vehicle = self.gate_associate_vehicle
        
        if z.shape[1] == 0:
            return
        else:
            for j in range(0, z.shape[1]):
                if z_class[j] == 'person':
                    gate_associate = gate_associate_person
                elif z_class[j] == 'car':
                    gate_associate = gate_associate_vehicle
                distance_m = float("inf")
                id_associate_best = float("inf")
                for k in range(0, int(xx.shape[0] / 4)):
                    xx_x = xx[4 * k, 0]
                    xx_y = xx[4 * k + 2, 0]
                    z_x = z[0, j]
                    z_y = z[1, j]
                    dd = (z_x - xx_x) ** 2 + (z_y - xx_y) ** 2
                    distance = math.sqrt(dd)
                    if distance < gate_associate and distance < distance_m:
                        distance_m = distance
                        id_associate_best = k
                # Association accomplished.
                if id_associate_best != float("inf"):
                    za_new = z[:, [j]]
                    za_cube_new = z_cube[:, range(3 * j, 3 * j + 3)]
                    za_class_new = z_class[j]
                    id_za_new = id_associate_best
                    if za.shape[1] == 0:
                        za = za_new
                        za_cube = za_cube_new
                        za_class = [za_class_new]
                        id_za = [id_za_new]
                    else:
                        za = np.column_stack((za, za_new))
                        za_cube = np.column_stack((za_cube, za_cube_new))
                        za_class.append(za_class_new)
                        id_za.append(id_za_new)
                    xx_mistracking[id_associate_best] = 0
                # Association failed.
                else:
                    zu_new = z[:, [j]]
                    zu_cube_new = z_cube[:, range(3 * j, 3 * j + 3)]
                    zu_class_new = z_class[j]
                    if zu.shape[1] == 0:
                        zu = zu_new
                        zu_cube = zu_cube_new
                        zu_class = [zu_class_new]
                    else:
                        zu = np.column_stack((zu, zu_new))
                        zu_cube = np.column_stack((zu_cube, zu_cube_new))
                        zu_class.append(zu_class_new)
            
            self.za = za
            self.za_cube = za_cube
            self.za_class = za_class
            self.id_za = id_za
            
            self.zu = zu
            self.zu_cube = zu_cube
            self.zu_class = zu_class
            
            self.xx_mistracking = xx_mistracking
            return
    
    def kf_update(self):
        """KF update.
        """
        
        za = self.za
        za_cube = self.za_cube
        za_class = self.za_class
        id_za = self.id_za
        
        xx = self.xx
        px = self.px
        xx_cube = self.xx_cube
        xx_class = self.xx_class
        
        r = self.noise_r
        
        if za.shape[1] == 0:
            return
        else:
            h = np.matrix([[1, 0, 0, 0],
                           [0, 0, 1, 0],])
            len_xx = xx.shape[0]
            len_za = za.shape[1]
            hh = np.zeros((2 * len_za, len_xx))
            zz = np.zeros((2 * len_za, 1))
            rr = np.zeros((2 * len_za, 2 * len_za))
            for j in range(0, len_za):
                ii = [2 * j, 2 * j + 1]
                jj = range(4 * id_za[j], 4 * id_za[j] + 4)
                for h_i in range(0, 2):
                    for h_j in range(0, 4):
                        hh[ii[h_i], jj[h_j]] = h[h_i, h_j]
                zz[ii, :] = za[:, [j]] - hh[ii, :] * xx
                for r_i in range(0, 2):
                    for r_j in range(0, 2):
                        rr[ii[r_i], ii[r_j]] = r[r_i, r_j]
            kk = px * hh.T * np.linalg.inv(hh * px * hh.T + rr)
            xx = xx + kk * zz
            px = px - kk * hh * px
            
            for j in range(0, len_za):
                xx_cube[range(8 * id_za[j], 8 * id_za[j] + 8), :] = za_cube[:, range(3 * j, 3 * j + 3)]
                xx_class[id_za[j]] = za_class[j]
            
            self.xx = xx
            self.px = px
            self.xx_cube = xx_cube
            self.xx_class = xx_class
            return
    
    def delete(self):
        """Delete targets which are beyond the range of observation.
        """
        
        xx = self.xx
        px = self.px
        xx_cube = self.xx_cube
        xx_class = self.xx_class
        
        xx_mistracking = self.xx_mistracking
        xx_color_idx = self.xx_color_idx
        xtrue = self.xtrue
        max_tracking_times = self.max_tracking_times
        
        if xx.shape[1] == 0:
            return
        else:
            # Judge whether some target should be deleted for its mistracking.
            k = 0
            while k < int(xx.shape[0] / 4):
                if xx_mistracking[k] > max_tracking_times:
                    len_xx = xx.shape[0]
                    if len_xx == 4:
                        xx = np.matrix([])
                        px = np.matrix([])
                        xx_cube = np.matrix([])
                        xx_class = []
                        xx_mistracking = []
                        xx_color_idx = []
                        break
                    else:
                        xx = np.delete(xx, range(4 * k, 4 * k + 4), axis=0)
                        px = np.delete(px, range(4 * k, 4 * k + 4), axis=0)
                        px = np.delete(px, range(4 * k, 4 * k + 4), axis=1)
                        xx_cube = np.delete(xx_cube, range(8 * k, 8 * k + 8), axis=0)
                        xx_class.pop(k)
                        xx_mistracking.pop(k)
                        xx_color_idx.pop(k)
                        continue
                k += 1
            
            self.xx = xx
            self.px = px
            self.xx_cube = xx_cube
            self.xx_class = xx_class
            
            self.xx_mistracking = xx_mistracking
            self.xx_color_idx = xx_color_idx
            return
    
    def pre_augment(self):
        """Prepare for KF augment.
            aug contains velocity information which is obtained using two frames of location of the same target.
        """
        
        zu = self.zu
        zu_cube = self.zu_cube
        zu_class = self.zu_class
        
        pre_aug = self.pre_aug
        pre_aug_cube = self.pre_aug_cube
        pre_aug_class = self.pre_aug_class
        
        self.aug = np.matrix([])
        self.aug_cube = np.matrix([])
        self.aug_class = []
        
        time_interval = self.time_interval
        gate_associate_person = self.gate_associate_person
        gate_associate_vehicle = self.gate_associate_vehicle
        
        aug = self.aug
        aug_cube = self.aug_cube
        aug_class = self.aug_class
        
        if zu.shape[1] == 0:
            return
        else:
            if pre_aug.shape[1] == 0:
                pre_aug = zu
                pre_aug_cube = zu_cube
                pre_aug_class = zu_class
                
                self.pre_aug = pre_aug
                self.pre_aug_cube = pre_aug_cube
                self.pre_aug_class = pre_aug_class
                
                self.aug = aug
                self.aug_cube = aug_cube
                self.aug_class = aug_class
                return
            else:
                for j in range(0, zu.shape[1]):
                    if zu_class[j] == 'person':
                        gate_associate = gate_associate_person
                    elif zu_class[j] == 'car':
                        gate_associate = gate_associate_vehicle
                    pre_distance_m = float("inf")
                    pre_id_associate_best = float("inf")
                    for k in range(0, pre_aug.shape[1]):
                        dd = (zu[0, j] - pre_aug[0, k]) ** 2 + (zu[1, j] - pre_aug[1, k]) ** 2
                        pre_distance = math.sqrt(dd)
                        if pre_distance < gate_associate and pre_distance < pre_distance_m:
                            pre_distance_m = pre_distance
                            pre_id_associate_best = k
                    if pre_id_associate_best != float("inf"):
                        aug_new = np.matrix([[zu[0, j]],
                                             [(zu[0, j] - pre_aug[0, pre_id_associate_best]) / time_interval],
                                             [zu[1, j]],
                                             [(zu[1, j] - pre_aug[1, pre_id_associate_best]) / time_interval],])
                        aug_cube_new = zu_cube[:, range(3 * j, 3 * j + 3)]
                        aug_class_new = zu_class[j]
                        if aug.shape[1] == 0:
                            aug = aug_new
                            aug_cube = aug_cube_new
                            aug_class = [aug_class_new]
                        else:
                            aug = np.column_stack((aug, aug_new))
                            aug_cube = np.column_stack((aug_cube, aug_cube_new))
                            aug_class.append(aug_class_new)
                
                pre_aug = zu
                pre_aug_cube = zu_cube
                pre_aug_class = zu_class
                
                self.pre_aug = pre_aug
                self.pre_aug_cube = pre_aug_cube
                self.pre_aug_class = pre_aug_class
                
                self.aug = aug
                self.aug_cube = aug_cube
                self.aug_class = aug_class
                return
    
    def kf_augment(self):
        """KF augment.
        """
        
        xx = self.xx
        px = self.px
        xx_cube = self.xx_cube
        xx_class = self.xx_class
        
        xx_mistracking = self.xx_mistracking
        xx_color_idx = self.xx_color_idx
        color_map = self.color_map
        
        aug = self.aug
        aug_cube = self.aug_cube
        aug_class = self.aug_class
        
        r = self.noise_r
        
        if aug.shape[1] == 0:
            return
        else:
            len_aug = aug.shape[1]
            for j in range(0, len_aug):
                xx_new = aug[:, j]
                xx_cube_new = aug_cube[:, range(3 * j, 3 * j + 3)]
                xx_class_new = aug_class[j]
                s = np.matrix([[1, 0],
                               [0, 0],
                               [0, 1],
                               [0, 0],])
                px_new = s * r * s.T
                mistracking_new = 0
                color_idx_new = np.random.randint(1, len(color_map))
                
                # Augment xx.
                if xx.shape[1] == 0:
                    xx = xx_new
                    xx_cube = xx_cube_new
                    xx_class = [xx_class_new]
                else:
                    xx = np.row_stack((xx, xx_new))
                    xx_cube = np.row_stack((xx_cube, xx_cube_new))
                    xx_class.append(xx_class_new)
                
                # Augment px.
                if px.shape[1] == 0:
                    px = px_new
                else:
                    len_px = px.shape[1]
                    ii = range(-4, 0)
                    px = np.row_stack((px, np.zeros((4, len_px))))
                    px = np.column_stack((px, np.zeros((len_px + 4, 4))))
                    for p_i in range(0, 4):
                        for p_j in range(0, 4):
                            px[ii[p_i], ii[p_j]] = px_new[p_i, p_j]
                
                xx_mistracking.append(mistracking_new)
                xx_color_idx.append(color_idx_new)
            
            self.xx = xx
            self.px = px
            self.xx_cube = xx_cube
            self.xx_class = xx_class
            
            self.xx_mistracking = xx_mistracking
            self.xx_color_idx = xx_color_idx
            return

    def kf_iterate(self, z, z_cube, z_class):
        self.kf_predict()
        self.increase_mistracking()
        self.associate(z, z_cube, z_class)
        self.kf_update()
        self.delete()
        self.pre_augment()
        self.kf_augment()

