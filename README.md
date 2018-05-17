# Using-test_net_mot.py-
Analysis of 'test_net_mot' partial code on MOT (in progress)

### 1. Code and function used. ###  
  * from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list: for cfg  
  * from datasets.factory import get_imdb: for imdb  
  * from fast_rcnn.test import test_net: for test  
  * from fast_rcnn.test_mot import test_net_mot: for test(default max_per_image=300)  
  * from datasets.mot import mot: mot path Structure  

* ### 2. Arguments ###
  * –gpu*: GPU id to use  
  * –def*: prototxt file defining the network  
  * –net**: model to test  
  * –cfg*: optional config file  
  * --wait: wait until net file exists  
  * –imdb**: dataset to test  
  * --comp: competition mode  
  * --set: set config keys  
  * --vis: visualize detections  
  * --num_dets: max number of detections per image(default=100)  
  (*: using always, **: using when test)  

* ### Another function ### 
  * Change data directory path: config.py-194 

### 4. Code ###
* from fast_rcnn.test_mot import test_net_mot
  ```python
  1.	def test_net_mot(net, imdb, max_per_image=300, thresh=0.05, vis=False):  
  2.	    """Test a Fast R-CNN network on an image database."""  
  3.	    num_images = len(imdb.image_index)  
  4.	    # all detections are collected into:  
  5.	    #    all_boxes[cls][image] = N x 5 array of detections in  
  6.	    #    (x1, y1, x2, y2, score)  
  7.	    all_boxes = [[[] for _ in xrange(num_images)]  
  8.	                 for _ in xrange(imdb.num_classes)]  
  9.	  
  10.	    output_dir = get_output_dir(imdb, net)  
  11.	  
  12.	    # timers  
  13.	    _t = {'im_detect' : Timer(), 'misc' : Timer()}  
  14.	  
  15.	    if not cfg.TEST.HAS_RPN:  
  16.	        roidb = imdb.roidb  
  17.	  
  18.	    # ----------------------------------------------------------------------  
  19.	    # Faster R-CNN Object Detection  
  20.	    # ----------------------------------------------------------------------  
  21.	  
  22.	    det_file = os.path.join(output_dir, 'detections_noNMS_300.pkl')  
  23.	    # det_file = os.path.join(output_dir, 'detections_NMS.pkl')  
  24.	  
  25.	  
  26.	    if not os.path.isfile(det_file):  
  27.	        for i in xrange(num_images):  
  28.	            # filter out any ground truth boxes  
  29.	            if cfg.TEST.HAS_RPN:  
  30.	                box_proposals = None  
  31.	            else:  
  32.	                # The roidb may contain ground-truth rois (for example, if the roidb  
  33.	                # comes from the training or val split). We only want to evaluate  
  34.	                # detection on the *non*-ground-truth rois. We select those the rois  
  35.	                # that have the gt_classes field set to 0, which means there's no  
  36.	                # ground truth.  
  37.	                box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0] 
  ```
  
  line 26~37: Work to filter out a rois without ground-truth(if it has, set it to ‘none’ and if it has gt_class set to
  0)
  
  ```python
  38.	  
  39.	            im = cv2.imread(imdb.image_path_at(i))  
  40.	            _t['im_detect'].tic()  
  41.	            scores, boxes = im_detect(net, im, box_proposals)  
  42.	            _t['im_detect'].toc()  
  43.	  
  44.	            _t['misc'].tic() 
  ```
  
  line 40~44: I don’t know what tic, toc function do 
  
  ```python
  45.	            # skip j = 0, because it's the background class  
  46.	            for j in xrange(1, imdb.num_classes):  
  47.	                inds = np.where(scores[:, j] > thresh)[0]  
  48.	                cls_scores = scores[inds, j]  
  49.	                cls_boxes = boxes[inds, j*4:(j+1)*4]  
  50.	                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \  
  51.	                    .astype(np.float32, copy=False)  
  52.	                # TODO: temporary do not use NMS for tracking  
  53.	                keep = nms(cls_dets, cfg.TEST.NMS)  
  54.	                cls_dets = cls_dets[keep, :]  
  55.	                if vis:  
  56.	                    vis_detections(im, imdb.classes[j], cls_dets)  
  57.	                all_boxes[j][i] = cls_dets  
  ```
  
  line 47: inds is the scores greater than thresh(=0.05) for each class  
  line 48: create new array ‘cls_scores’ that is score for each class  
  line 50: cls_boxes, cls_scores into one array and change the type into float 32  
  line 53: Dispatch to either CPU or GPU NMS implementations.( from fast_rcnn.nms_wrapper import nms)  
  line 54: change the value of cls_dets origin to an array using keep
  
  ```python
  58.	            # Limit to max_per_image detections *over all classes*  
  59.	            if max_per_image > 0:  
  60.	                image_scores = np.hstack([all_boxes[j][i][:, -1]  
  61.	                                          for j in xrange(1, imdb.num_classes)])  
  62.	                if len(image_scores) > max_per_image:  
  63.	                    image_thresh = np.sort(image_scores)[-max_per_image]  
  64.	                    for j in xrange(1, imdb.num_classes):  
  65.	                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]  
  66.	                        all_boxes[j][i] = all_boxes[j][i][keep, :]  
  67.	            _t['misc'].toc()  
  68.	  
  69.	            print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \  
  70.	                  .format(i + 1, num_images, _t['im_detect'].average_time,  
  71.	                          _t['misc'].average_time) 
  ```

  line 59\~61: if max_per image(300) is greater than 0, image_scores is 클래스의 수만큼의 all_boxs에대한 마지막 값(?)  
  line 62\-65: if image_scores’s lengh is greater than 300, image_thresh is front of sorted image_scores to 300th value.  
  line 66: insert new array that contain keep in array of all_boxes  

  ```python
  72.	        # save detection results  
  73.	        with open(det_file, 'wb') as f:  
  74.	            cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)  
  75.	  
  76.	    else:  # if result already exist, load detection results  
  77.	        with open(det_file, 'rb') as f:  
  78.	            all_boxes = np.array(cPickle.load(f))  
  79.	  
  80.	    # load all image's path  
  81.	    image_list = []  
  82.	    for i in xrange(num_images):  
  83.	        image_list.append(imdb.image_path_at(i))  
  84.	  
  85.	    # extract video frame number  
  86.	    vid_frame = range(num_images)  
  87.	    for i in xrange(num_images):  
  88.	        vid_frame[i] = int(image_list[i].split('/')[-1].split('.')[0])  
  89.	  
  90.	    time_start = time.time()  
  91.	  
  92.	    output_dir = os.path.join('./output', imdb.name + '_result')  
  93.	  
  94.	    video_start_index = 0  
  95.	  
  96.	    # split detections by video  
  97.	    cnt = 0  
  98.	    for im_ind in xrange(num_images):  
  99.	        if vid_frame[im_ind] == 1:  # video start point  
  100.	            video_start_index = im_ind  
  101.	        elif im_ind == num_images - 1 or vid_frame[im_ind] > vid_frame[im_ind + 1]:  # video end point  
  102.	            if cnt != 1:  
  103.	  
  104.	                video_len = im_ind - video_start_index + 1  
  105.	                video_image_list = image_list[video_start_index:im_ind+1]  
  106.	                # get this video's detections  
  107.	                video_all_boxes = [[[] for _ in xrange(video_len)] for _ in xrange(imdb.num_classes)]  
  108.	                for cls in xrange(1, imdb.num_classes):  
  109.	                    video_all_boxes[cls] = all_boxes[cls][video_start_index:im_ind+1]  
  110.	  
  111.	                # perform multi-object tracking within video  
  112.	                # tracked_all_boxes = mcmot(video_all_boxes, video_image_list, conf_thresh=0.3)  
  113.	                tracked_all_boxes = mcmot_cpd(video_all_boxes, video_image_list, conf_thresh=0.3)  
  114.	                # tracked_all_boxes = mcmot_conference(video_all_boxes, video_image_list, conf_thresh=0.3)  
  ```
  
  line 113: return tracked boxes(from tracking.multi_object_tracking_cpd import mcmot_cpd) a detailed explanation will be given below.
  
  ```python
  116.	                for cls in xrange(1, imdb.num_classes):  
  117.	                    all_boxes[cls][video_start_index:im_ind+1] = tracked_all_boxes[cls]  
  118.	  
  119.	                print 'perform multi-object tracking per video {:d}/{:d}' .format(im_ind, num_images)  
  120.	  
  121.	                # # Trajectory Visualization  
  122.	                #class_name = imdb.classes  
  123.	                #vis_trajectories(class_name, video_image_list, tracked_all_boxes, thresh=0.7, sec=0.05)  
  124.	  
  125.	                # TODO: change thresholding, such as, keep high score trajectory but also long trajectory with low score  
  126.	  
  127.	                # mot format: frame num, tracking id, x, y, width, height, confidence score, -1, -1, -1  
  128.	                # generate mot format result file  
  129.	                mot_result = []  
  130.	                score_thresh = 0.5 # DK180102 0.7 -> 0.5  
  131.	                for frame, i in enumerate(range(video_start_index, im_ind+1)):  
  132.	                    dets = all_boxes[1][i]  
  133.	                    for j in xrange(len(dets)):  
  134.	                        det = dets[j]  
  135.	                        if det[4] < score_thresh:  
  136.	                            continue  
  137.	                        det_mot = [frame + 1, det[5] + 1, det[0] + 1.0, det[1] + 1.0, det[2] - det[0], det[3] - det[1], det[4], -1, -1,  
  138.	                                   -1]  
  139.	                        mot_result.append(det_mot)  
  140.	  
  141.	                if not os.path.isdir(output_dir):  
  142.	                    os.makedirs(output_dir)  
  143.	                result_file_name = os.path.join(output_dir, video_image_list[0].split('/')[-3] + '.txt')  
  144.	                np.savetxt(result_file_name, mot_result, delimiter=',',  
  145.	                           fmt=('%d', '%d', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%d', '%d', '%d'))  
  146.	            cnt += 1  
  147.	  
  148.	  
  149.	    time_end = time.time()  
  150.	    tracking_time = time_end - time_start  
  151.	    print 'Total tracking time is : {:f}'.format(tracking_time)  
  152.	  
  153.	    # EK20160428 Run Matlab code  
  154.	    # Evaluation  
  155.	  
  156.	    eval_dir = '/home/sdk1/data/MOT17Det/motchallenge-devkit/motchallenge'  # MOT evaluation file path  
  157.	    output_dir_full = os.path.join(os.getcwd(), output_dir)  
  158.	  
  159.	    cmd = 'cd {} && '.format(eval_dir)  
  160.	    cmd += 'matlab -nodisplay -nodesktop '  
  161.	    cmd += '-r "dbstop if error; '  
  162.	    # cmd += 'evaluateTracking(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \  
  163.	    #     .format('c2-train.txt', output_dir, image_dir)  # MOT2015 train  
  164.	    cmd += 'evaluateTracking(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \  
  165.	        .format('c10-train.txt', output_dir_full, '/home/sdk1/data/MOT/MOT2016/train/')  # MOT2016 train  
  166.	    print('Running:\n{}'.format(cmd))  
  167.	    status = subprocess.call(cmd, shell=True)  
  ```
  
  line 129~151: code for visualization and timer  
  line 156~166: code for evaluation  

  * from tracking.multi_object_tracking_cpd import mcmot_cpd(tracking code)
  ```python
  1.	def mcmot_cpd(all_boxes, image_list, conf_thresh=0.5):  
  2.	    """ 
  3.	    Multi-class Multi-object Tracking 
  4.	 
  5.	    Args: 
  6.	        all_boxes: float numpy array, shape(n_class, n_frames, n_regions, 5), Faster R-CNN detection result format 
  7.	                each column means (x1, y1, x2, y2, confidence) 
  8.	        image_list: string array, shape(n_frames), image location for each frame, same index with all_boxes 
  9.	        conf_thresh: float, detection confidence threshold 
  10.	 
  11.	    Returns: 
  12.	        tracked_boxes: float numpy array, shape(n_class, n_frames, n_regions, 6), multi-object tracking result 
  13.	                each column means (x1, y1, x2, y2, confidence, tracking ID) 
  14.	    """  
  15.	  
  16.	    # fix the random seeds for reproducibility  
  17.	    np.random.seed(3)  
  18.	  
  19.	    # initialization  
  20.	    num_classes = len(all_boxes)  
  21.	    num_images = len(all_boxes[0])  
  22.	    im_shape = cv2.imread(image_list[0]).shape  
  23.	    tracked_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(num_classes)]  
  24.	    all_boxes = np.array(all_boxes)  
  25.	    vid_frame = range(num_images)  
  26.	  
  27.	    # record running time  
  28.	    time_start = time.time()  
  29.	  
  30.	    for cls_ind in xrange(1, num_classes):  
  31.	        # initialization for each class  
  32.	        trajectories = []  # list of all trajectories  
  33.	        count_track_id = 0  # identity allocator for trajectory generation  
  34.	        activations = []  # index of activated trajectories  
  35.	  
  36.	        for im_ind in xrange(num_images):  
  37.	            # load Faster R-CNN's object detection result for this class  
  38.	            dets = all_boxes[cls_ind][im_ind]  
  39.	  
  40.	            # filter strange ratio box (especially for pedestrian)  
  41.	            # TODO: if your dataset include not only person, please do not use 'person_ratio_check' function  
  42.	            dets = person_ratio_cehck(dets, ratio_param=1.5)  
  ```
  
  line 42: use person_ratio_check only if it include only person(but in code it is ‘cehck’. maybe it’s a typo)  
       (return: filtered detection results from input dets using width, height ratio)  
       +) I don’t know what to use for dataset including person  
       from utilities import person_ratio_cehck  

  ```python
  43.	            # calculate color histogram on each detection response  
  44.	            color_feats = np.array(calc_color_histogram(image_list[im_ind], dets[:, 0:4], bin_size=180))  
  45.	  
  46.	            # apply NMS on detection to get candidate state's position  
  47.	            nms_thresh1 = 0.4  
  48.	            nms_thresh2 = 0.7  
  49.	            keep, keep_group = nms(dets, nms_thresh1, nms_thresh2)  
  ```
  line 46: NMS(Non-Maximum Suppression) from utilities import nms
  ```python
  50.	            # only keep confident NMS detection  
  51.	            conf_filter = np.where(dets[keep, 4] > conf_thresh)[0]  
  52.	            keep = np.array(keep)[conf_filter]  
  53.	            keep_group = np.array(keep_group)[conf_filter]  
  54.	  
  55.	            if trajectories == [] and dets != []:  
  56.	                # init frame, first time to generate trajectory  
  57.	  
  58.	                for i in xrange(len(keep)):  
  59.	                    # get one group of NMS result  
  60.	                    inds = np.append(keep[i], keep_group[i])  
  61.	                    # generate new trajectory (birth)  
  62.	                    trajectories.append(trajectory(vid_frame[im_ind], count_track_id, dets[inds], color_feats[inds]))  
  63.	                    # put activated ID in list  
  64.	                    activations.append(count_track_id)  
  65.	                    # update ID allocator  
  66.	                    count_track_id += 1  
  67.	  
  68.	            elif dets == []:  
  69.	                # this frame doesn't have any proposals, decrease all trajectories's state_conf  
  70.	                if vid_frame[im_ind] != 0 and activations != 0:  
  71.	                    for i in activations:  
  72.	                        state = trajectories[i].update(vid_frame[im_ind], [], [])  
  73.	                        if state == 'death':  
  74.	                            activations.remove(i)  
  75.	  
  76.	            else:  
  77.	                # middle of video frame, perform tracking  
  78.	  
  79.	                # predict (posterior at t-1) -> motion model -> measure (likelihood) -> correct (posterior at t)  
  80.	  
  81.	                index = np.array(activations)  
  82.	  
  83.	                # shuffle index, randomly choose targets with uniform probability for stochastic process  
  84.	                np.random.shuffle(index)  
  85.	  
  86.	                print '{:d} / {:d}'.format(im_ind, num_images)  
  87.	  
  88.	                used_det_index = []  
  89.	  
  90.	                for j, i in enumerate(index):  
  91.	                    # "one target at a time" scheme in [Khan05]  
  92.	                    # select a single target m, chosen from all targets with uniform probability,  
  93.	                    # and updates its state X_tm only by sampling from a single target proposal density.  
  94.	  
  95.	                    # ---------------------------------------------  
  96.	                    # apply motion model, estimate particles  
  97.	                    # ---------------------------------------------  
  98.	  
  99.	                    # estimate particle by applying motion model on posterior estimated at t-1  
  100.	                    est_particle = trajectories[i].get_est_motion_particle()  
  101.	  
  102.	                    # check out of coordinate for each particle  
  103.	                    for p in xrange(len(est_particle)):  
  104.	                        est_particle[p] = np.array(check_out_of_image(est_particle[p], [im_shape[1], im_shape[0]]))  
  105.	  
  106.	                    # Data-driven proposal probabilities are used to direct the Markov chain dynamics  
  107.	                    # taking proposals using overlap can be considered as diffusing  
  108.	                    # by applying noise to spread particles  
  109.	  
  110.	                    # calculate box overlap between particles and detections  
  111.	                    overlaps = utils.cython_bbox.bbox_overlaps(np.array(est_particle).astype(np.float),  
  112.	                                                               np.array(dets[:, 0:4]).astype(np.float))  
  113.	  
  114.	                    # get particles from proposal which overlap over threshold  
  115.	                    overlap_thresh = 0.7  
  116.	                    overlap_index = np.where(overlaps > overlap_thresh)  
  117.	                    unique_det_index = np.unique(overlap_index[1])  
  118.	  
  119.	                    # exclude used particles  
  120.	                    unique_det_index = [x for x in unique_det_index if x not in used_det_index]  
  121.	                    est_particle = dets[unique_det_index, :]  
  122.	                    est_particle_det = est_particle[:, 4]  
  123.	  
  124.	                    used_det_index.extend(unique_det_index)  
  125.	  
  126.	                    # ------------------------------------------------  
  127.	                    # observation - color histogram, calc likelihood  
  128.	                    # ------------------------------------------------  
  129.	  
  130.	                    # calc color histogram per each particle  
  131.	                    est_particle_color_hist = color_feats[unique_det_index, :]  
  132.	  
  133.	                    # calc Bhattacharyya distance to get similarity, 0~1, 1=perfect match, 0=mismatch  
  134.	                    est_particle_color_sim = calc_hist_similarity(trajectories[i].get_color_hist(),  
  135.	                                                                  est_particle_color_hist)  
  136.	  
  137.	                    # weights are assigned by likelihood response  
  138.	                    w_ratio = 0.5  
  139.	                    est_particle_weight = (1.0 - w_ratio) * est_particle_color_sim + w_ratio * est_particle_det  
  140.	                    est_particle[:, 4] = est_particle_weight  
  141.	  
  142.	                    # TODO: if particle weights are lower than threshold, do re-sampling  
  143.	                    # TODO: if particles are less centerized, do re-sampling  
  144.	  
  145.	                    # ------------------------------------------------  
  146.	                    # update  
  147.	                    # ------------------------------------------------  
  148.	  
  149.	                    state = trajectories[i].update(vid_frame[im_ind], est_particle, est_particle_color_hist, est_particle_color_sim, est_particle_det)  
  150.	                    # check state transaction from 'track' to 'death'  
  151.	                    if state == 'death':  
  152.	                        activations.remove(i)  
  153.	  
  154.	                # TODO: check duplicated trajectories by using temporal overlap  
  155.	  
  156.	                # ------------------------------------------------------  
  157.	                # state transaction, 'birth' and 're-birth'  
  158.	                # ------------------------------------------------------  
  159.	  
  160.	                if len(activations) > 0:  
  161.	  
  162.	                    # get detection nms result  
  163.	                    det_boxes = dets[keep, 0:4]  
  164.	  
  165.	                    # load activated trajectories's state  
  166.	                    trajectory_boxes = []  
  167.	                    for i in activations:  
  168.	                        trajectory_boxes.append(trajectories[i].get_last_box())  
  169.	  
  170.	                    # calculate box overlap between trajectories and detections  
  171.	                    overlaps = utils.cython_bbox.bbox_overlaps(np.array(det_boxes).astype(np.float),  
  172.	                                                               np.array(trajectory_boxes).astype(np.float))  
  173.	  
  linw 160~168: 
  174.	                    # find not overlapped detection which consider as new trajectory  
  175.	                    overlap_index = np.unique(np.where(overlaps > 0.3)[0])  
  176.	                    det_index = range(0, len(det_boxes))  
  177.	                    if len(overlap_index) != len(det_index):  # if there is new trajectory  
  178.	  
  179.	                        # find detections which are for birth trajectories  
  180.	                        birth_index = [x for x in det_index if x not in overlap_index]  
  181.	  
  182.	                        # shuffle index for stochastic process  
  183.	                        np.random.shuffle(birth_index)  
  184.	  
  185.	                        # get recent 'death' trajectories to check re-birth  
  186.	                        death_index = []  
  187.	                        death_color_hists = []  
  188.	                        death_boxes = []  
  189.	                        used_death_index = []  
  190.	                        for ind, t in enumerate(trajectories):  
  191.	                            if t.get_state() == 'death' and 0 < (vid_frame[im_ind] - t.get_frame()) < 30 and t.tracked:  
  192.	                                death_index.append(ind)  
  193.	                                death_color_hists.append(t.get_color_hist())  
  194.	                                death_boxes.append(t.get_last_box())  
  195.	  
  196.	                        # for each birth candidates, check 're-birth'  
  197.	                        for ind, i in enumerate(birth_index):  
  198.	                            inds = np.append(keep[i], keep_group[i])  
  199.	                            birth_dets = dets[inds]  
  200.	                            birth_color_feats = color_feats[inds]  
  201.	                            re_birth = False  
  ```
  
  line ~201: chech death for re birth and birth  

  ```python
  202.	                            # ------------------------------------  
  203.	                            # 're-birth' state transaction  
  204.	                            # ------------------------------------  
  205.	  
  206.	                            if len(death_index) > 0:  
  207.	  
  208.	                                birth_weights = birth_dets[:, 4] / np.sum(birth_dets[:, 4])  
  209.	                                birth_box = [weighted_sum(birth_dets[:, 0:4], birth_weights)]  
  210.	                                birth_color_feat = weighted_sum(birth_color_feats, birth_weights)  
  211.	  
  212.	                                # check color similarity between new detection and death trajectories  
  213.	                                birth_color_sim = calc_hist_similarity(birth_color_feat, death_color_hists)  
  214.	  
  215.	                                # check overlap between new detection and death trajectories  
  216.	                                birth_overlaps = \  
  217.	                                utils.cython_bbox.bbox_overlaps(np.array(birth_box).astype(np.float16).astype(np.float),  
  218.	                                                                np.array(death_boxes).astype(np.float16).astype(np.float))[  
  219.	                                    0]  
  220.	  
  221.	                                # calc rebirth confidence for each match and find maximum confidence  
  222.	                                rebirth_conf = (birth_color_sim + birth_overlaps) / 2  
  223.	                                rebirth_conf_max_index = np.argmax(rebirth_conf)  
  224.	  
  225.	                                if rebirth_conf[rebirth_conf_max_index] > 0.4:  # do re-birth  
  226.	                                    rebirth_index = death_index[rebirth_conf_max_index]  
  227.	                                    if rebirth_index not in used_death_index:  
  228.	                                        death_box = death_boxes[rebirth_conf_max_index]  
  229.	  
  230.	                                        # set searching range  
  231.	                                        r = (death_box[2] - death_box[0] + death_box[3] - death_box[1]) / 2  
  232.	  
  233.	                                        # check is distance close between re-birth candidates  
  234.	                                        if birth_box[0][0] > death_box[0] - r and birth_box[0][1] > death_box[1] - r and \  
  235.	                                        birth_box[0][2] < death_box[2] + r and birth_box[0][3] < death_box[3] + r:  
  236.	  
  237.	                                            # get connected ID  
  238.	                                            rebirth_id = trajectories[rebirth_index].get_tid()  
  239.	  
  240.	                                            # generate new trajectory with exist ID (re-birth)  
  241.	                                            trajectories[rebirth_id].re_birth(vid_frame[im_ind], rebirth_id, birth_dets, birth_color_feats)  
  242.	                                            activations.append(rebirth_id)  
  243.	                                            used_death_index.append(rebirth_index)  
  244.	                                            re_birth = True  
  245.	  
  246.	                            if not re_birth:  
  247.	  
  248.	                                # generate new trajectory (birth)  
  249.	                                trajectories.append(  
  250.	                                    trajectory(vid_frame[im_ind], count_track_id, birth_dets, birth_color_feats))  
  251.	  
  252.	                                # put activated ID in list  
  253.	                                activations.append(count_track_id)  
  254.	  
  255.	                                # update ID allocator  
  256.	                                count_track_id += 1  
  257.	  
  258.	                elif len(activations) == 0:  
  259.	                    # there is no activations, all observed detections are considered as 'birth' state  
  260.	                    for i in xrange(len(keep)):  
  261.	                        inds = np.append(keep[i], keep_group[i])  
  262.	                        birth_dets = dets[inds]  
  263.	                        birth_color_feats = color_feats[inds]  
  264.	  
  265.	                        # generate new trajectory (birth)  
  266.	                        trajectories.append(trajectory(vid_frame[im_ind], count_track_id, birth_dets, birth_color_feats))  
  267.	  
  268.	                        # put activated ID in list  
  269.	                        activations.append(count_track_id)  
  270.	  
  271.	                        # update ID allocator  
  272.	                        count_track_id += 1  
  273.	  
  274.	        # -------------------------------------------  
  275.	        # post processing after tracking  
  276.	        # -------------------------------------------  
  277.	  
  278.	        tracked_trajectories_len = []  
  279.	        tracked_trajectories_idx = []  
  280.	        for idx in xrange(len(trajectories)):  
  281.	            if trajectories[idx].tracked:  
  282.	                tracked_trajectories_idx.append(idx)  
  283.	                tracked_trajectories_len.append(len(trajectories[idx].tracking_confidence))  
  284.	  
  285.	        sorted_idx = sorted(range(len(tracked_trajectories_len)), key=lambda x:tracked_trajectories_len[x], reverse=True)  
  286.	        tracked_trajectories = np.array(trajectories)[np.array(tracked_trajectories_idx)[sorted_idx]]  
  287.	  
  288.	        # load GroundTruth information  
  289.	        gt_path = os.path.join(os.path.split(os.path.split(image_list[0])[0])[0], 'gt', 'gt.txt')  
  290.	        annotations = np.loadtxt(gt_path, delimiter=',')  
  291.	  
  292.	        gt_annotation_idx = []  
  293.	        for frame in xrange(num_images):  
  294.	            # only sample visible pedestrian  
  295.	            anno_inds = np.where(annotations[:, 0] == frame + 1)[0]  
  296.	            confidence_inds = np.where(annotations[anno_inds, 6] == 1)[0]  
  297.	            pedestrian_inds = np.where(annotations[anno_inds, 7] == 1)[0]  
  298.	            visibility_inds = np.where(annotations[anno_inds, 8] >= 0.1)[0]  
  299.	  
  300.	            gt_inds = list(set(confidence_inds) & set(pedestrian_inds) & set(visibility_inds))  
  301.	            gt_inds = anno_inds[gt_inds]  
  302.	            gt_annotation_idx.append(gt_inds)  
  303.	  
  304.	        # CPD analysis  
  305.	        for traj in tracked_trajectories:  
  306.	            # detection responses  
  307.	            cpd_dets = []  
  308.	            for i in xrange(len(traj.all_dets)):  
  309.	                cpd_dets.append(np.mean(traj.all_dets[i]))  
  310.	  
  311.	            # appearance  
  312.	            cpd_appearance = []  
  313.	            for i in xrange(len(traj.all_color_sims)):  
  314.	                cpd_appearance.append(np.mean(traj.all_color_sims[i]))  
  315.	  
  316.	            # xywh  
  317.	            cpd_xywh = np.array(traj.diff_xywh)  
  318.	  
  319.	            # tracking confidence  
  320.	            cpd_tracking_confidence = traj.tracking_confidence  
  321.	  
  322.	            # gt overlaps  
  323.	            gt_overlaps = []  
  324.	  
  325.	            # swap  
  326.	            swap = []  
  327.	  
  328.	            traj_id = 0  
  329.	            for box in traj.all_boxes:  
  330.	                frame = int(box[0])  
  331.	                gt_ids = annotations[gt_annotation_idx[frame], 1]  
  332.	                gt_boxes = annotations[gt_annotation_idx[frame], 2:6]  
  333.	                gt_boxes[:, 2] += gt_boxes[:, 0]  
  334.	                gt_boxes[:, 3] += gt_boxes[:, 1]  
  335.	                for id, b in enumerate(gt_boxes):  
  336.	                    if b[0] < 0:  
  337.	                        gt_boxes[id][0] = 0  
  338.	                    if b[1] < 0:  
  339.	                        gt_boxes[id][0] = 0  
  340.	  
  341.	                # check overlap between trajectory box and gt boxes  
  342.	                overlaps = utils.cython_bbox.bbox_overlaps(np.array([box[1:5]]), gt_boxes)[0]  
  343.	                max_idx = np.argmax(overlaps)  
  344.	                gt_overlaps.append(overlaps[max_idx])  
  345.	                if overlaps[max_idx] > 0.3:  
  346.	                    if frame == 0:  
  347.	                        traj_id = gt_ids[max_idx]  
  348.	                        swap.append(0)  
  349.	                    else:  
  350.	                        if traj_id == gt_ids[max_idx]:  
  351.	                            swap.append(0)  
  352.	                        else:  
  353.	                            traj_id = gt_ids[max_idx]  
  354.	                            swap.append(1)  
  355.	                else:  
  356.	                    swap.append(0)  
  357.	  
  358.	        print 'debug'  
  359.	  
  360.	  
  361.	  
  362.	        for i in trajectories:  
  363.	            if i.tracked:  # if it is tracked trajectory  
  364.	                i.all_boxes = np.array(i.all_boxes)  
  365.	                frames = i.all_boxes[:, 0].astype(int)  
  366.	                avg_score = np.sum(i.tracking_confidence) / len(i.tracking_confidence)  
  367.	  
  368.	                for index, frame in enumerate(frames):  
  369.	                    tracked_boxes[cls_ind][int(frame)].append(  
  370.	                        np.append(i.all_boxes[index][1:5], [avg_score, i.get_tid()]))  
  371.	  
  372.	    # convert to numpy array  
  373.	    for cls_ind in xrange(1, num_classes):  
  374.	        for im_ind in xrange(num_images):  
  375.	            if tracked_boxes[cls_ind][im_ind] != []:  
  376.	                tracked_boxes[cls_ind][im_ind] = np.array(tracked_boxes[cls_ind][im_ind], dtype=np.float32)  
  377.	  
  378.	    time_end = time.time()  
  379.	    tracking_time = time_end - time_start  
  380.	    print 'Tracking time is : {:f}'.format(tracking_time)  
  381.	  
  382.	    return tracked_boxes  
  ```
  
  * from trajectory_cpd import trajectory
  ```python
  1.	class trajectory:  
  2.	    """ 
  3.	    Each trajectory is presented by ID and coordinate 
  4.	    Trajectory mainly contains tracking ID, particles, appearance model, motion model. 
  5.	 
  6.	    Tracking ID means trajectory's unique identity 
  7.	    Particle means weighted samples which can represent state's distribution in 'Particle Filter' manner 
  8.	    Appearance model means that can estimate observation likelihood based on previous observation 
  9.	    Motion model means that can estimate dynamic motion based on previous observation 
  10.	    """  
  11.	  
  12.	    def __init__(self, frame, tid, boxes, color_feats):  # create trajectory as 'birth' state  
  13.	        self.state = 'birth'  # state {birth, track, death}  
  14.	        self.tracked = False  # flag to check is it tracked or only birth  
  15.	        self.all_particles = [boxes[:, 0:5]]  # [x1, y1, x2, y2, weights]  
  16.	        self.weights = [boxes[:, 4] / np.sum(boxes[:, 4])]  
  17.	        box = weighted_sum(boxes[:, 0:4], self.get_last_weights())  # point estimation by computing the marginal mean  
  18.	        self.all_boxes = [np.append(frame, box)]  # [frame_num, x1, y1, x2, y2, score]  
  19.	        self.tid = tid  # tracking Identity (ID)  
  20.	        self.frame = frame  # last frame's number  
  21.	        # appearance (color) model  
  22.	        self.color_model = weighted_sum(color_feats, self.get_last_weights())  
  23.	        # Constant-velocity linear motion model, (vx, vy, vw, vh)  
  24.	        self.motion_model = np.array([0., 0., 0., 0.])  
  25.	        # initialize tracking confidence by weighted sum of detection confidence  
  26.	        self.tracking_confidence = [np.sum(boxes[:, 4] * self.get_last_weights().T)]  
  27.	  
  28.	        # for CPD analysis  
  29.	        self.all_dets = [boxes[:, 4]]  
  30.	        self.all_color_sims = [np.ones(len(boxes), np.float64)]  
  31.	        self.diff_xywh = [np.zeros(4, np.float64)]  
  32.	        self.cpd_model = [changefinder.ChangeFinder(r=0.01, order=1, smooth=5)] * 6  
  33.	        self.cpd_score = [[] for _ in xrange(6)]  
  34.	        self.cpd_score[0] = self.cpd_model[0].update(np.mean(self.all_dets[-1]))  # detection  
  35.	        self.cpd_score[1] = self.cpd_model[1].update(np.mean(self.all_color_sims[-1]))  # appearance  
  36.	        self.cpd_score[2] = self.cpd_model[2].update(self.diff_xywh[-1][0])  # x difference  
  37.	        self.cpd_score[3] = self.cpd_model[3].update(self.diff_xywh[-1][1])  # y difference  
  38.	        self.cpd_score[4] = self.cpd_model[4].update(self.diff_xywh[-1][2])  # w difference  
  39.	        self.cpd_score[5] = self.cpd_model[5].update(self.diff_xywh[-1][3])  # h difference  
  40.	  
  41.	    def update(self, frame, particle, color_feats, color_sim, det, lr=0.5, particle_num_thresh=10):  
  42.	        """ 
  43.	        'update' state transaction. update trajectory by estimation and observation. 
  44.	        Args: 
  45.	            frame: int, indicate frame number 
  46.	            particle: float numpy array, shape (n_particles, 5), number of particles to update 
  47.	                    each column means (x0, y0, x1, y1, weight) 
  48.	            color_feats: float numpy array, shape (n_particles, color_histogram_dimension) 
  49.	                        color histogram feature extracted from each particle 
  50.	            lr: float, learning rate for update 
  51.	            particle_num_thresh: maximum number of particles to keep per each object 
  52.	                                if observation is robust, small number of particles is enough to track 
  53.	 
  54.	        Returns: 
  55.	            self.state: str, {'birth', 'track', 'death'}, current state of this trajectory 
  56.	        """  
  57.	  
  58.	        if len(particle) > particle_num_thresh:  
  59.	            # sort particles ordered by confidence  
  60.	            sorted_index = np.argsort(particle[:, 4])[::-1]  
  61.	            # only keep high confident particles by threshold  
  62.	            particle = particle[sorted_index[0:particle_num_thresh]]  
  63.	            color_feats = color_feats[sorted_index[0:particle_num_thresh]]  
  64.	            color_sim = color_sim[sorted_index[0:particle_num_thresh]]  
  65.	            det = det[sorted_index[0:particle_num_thresh]]  
  66.	  
  67.	        if len(particle) == 0 and len(color_feats) == 0:  
  68.	            # if there is no observation  
  69.	  
  70.	            # observation is replaced by estimation  
  71.	            est_particle = self.get_est_motion_particle()  
  72.	            est_box = self.get_est_motion_box()  
  73.	  
  74.	            # decrease all estimated particle's confidence as half since there is no observation  
  75.	            est_particle[:, 4] = est_particle[:, 4] / 2  
  76.	  
  77.	            # get t-1 weights  
  78.	            weights = self.get_last_weights()  
  79.	  
  80.	            # calculate tracking confidence  
  81.	            est_tracking_confidence = np.sum(est_particle[:, 4] * weights.T)  
  82.	  
  83.	            # set estimated present state  
  84.	            self.all_particles.append(np.array(est_particle))  
  85.	            self.all_boxes.append(np.append(frame, est_box))  
  86.	            self.weights.append(weights)  
  87.	            self.frame = frame  
  88.	            self.tracking_confidence.append((self.get_last_tracking_confidence() + est_tracking_confidence) / 2)  
  89.	  
  90.	            # for CPD analysis  
  91.	            self.all_dets.append(np.zeros(len(est_box), np.float32))  
  92.	            self.all_color_sims.append(np.zeros(len(est_box), np.float64))  
  93.	            self.diff_xywh.append(np.zeros(4, np.float64))  
  94.	  
  95.	            _ = self.cpd_model[0].update(np.mean(self.all_dets[-1]))  # detection  
  96.	            _ = self.cpd_model[1].update(np.mean(self.all_color_sims[-1]))  # appearance  
  97.	            _ = self.cpd_model[2].update(self.diff_xywh[-1][0])  # x difference  
  98.	            _ = self.cpd_model[3].update(self.diff_xywh[-1][1])  # y difference  
  99.	            _ = self.cpd_model[4].update(self.diff_xywh[-1][2])  # w difference  
  100.	            _ = self.cpd_model[5].update(self.diff_xywh[-1][3])  # h difference  
  101.	  
  102.	        else:  
  103.	            # weight normalization  
  104.	            weights = particle[:, 4] / np.sum(particle[:, 4])  
  105.	            obs_box = weighted_sum(particle[:, 0:4], weights)  
  106.	  
  107.	            # tracking confidence  
  108.	            obs_tracking_confidence = np.sum(particle[:, 4] * weights.T)  
  109.	  
  110.	            # motion model update  
  111.	            motion_residual = self.calc_motion_residual(obs_box)  
  112.	  
  113.	            # appearance model update (color histogram)  
  114.	            obs_color_hist = weighted_sum(color_feats, weights)  
  115.	  
  116.	            # for CPD analysis  
  117.	            self.all_color_sims.append(color_sim)  
  118.	            self.all_dets.append(det)  
  119.	            self.diff_xywh.append(motion_residual)  
  120.	  
  121.	            # # CPD model update  
  122.	            # cpd_average_score = 0  
  123.	            # cpd_average_score += self.cpd_model[0].update(np.mean(self.all_dets[-1]))  # detection  
  124.	            # cpd_average_score += self.cpd_model[1].update(np.mean(self.all_color_sims[-1]))  # appearance  
  125.	            # cpd_average_score += self.cpd_model[2].update(self.diff_xywh[-1][0])  # x difference  
  126.	            # cpd_average_score += self.cpd_model[3].update(self.diff_xywh[-1][1])  # y difference  
  127.	            # cpd_average_score += self.cpd_model[4].update(self.diff_xywh[-1][2])  # w difference  
  128.	            # cpd_average_score += self.cpd_model[5].update(self.diff_xywh[-1][3])  # h difference  
  129.	            # cpd_average_score = cpd_average_score / 6  
  130.	            #  
  131.	            cpd_average_score = [0.0] * 6  
  132.	            cpd_average_score[0] = self.cpd_model[0].update(np.mean(self.all_dets[-1]))  # detection  
  133.	            cpd_average_score[1] = self.cpd_model[1].update(np.mean(self.all_color_sims[-1]))  # appearance  
  134.	            cpd_average_score[2] = self.cpd_model[2].update(self.diff_xywh[-1][0])  # x difference  
  135.	            cpd_average_score[3] = self.cpd_model[3].update(self.diff_xywh[-1][1])  # y difference  
  136.	            cpd_average_score[4] = self.cpd_model[4].update(self.diff_xywh[-1][2])  # w difference  
  137.	            cpd_average_score[5] = self.cpd_model[5].update(self.diff_xywh[-1][3])  # h difference  
  138.	            cpd_average_score = np.max(cpd_average_score) / 100  
  139.	  
  140.	            if cpd_average_score > 1.0:  
  141.	                cpd_average_score = 1.0  
  142.	            elif cpd_average_score < 0:  
  143.	                cpd_average_score = 0  
  144.	  
  145.	            if len(self.all_dets) > 10:  
  146.	                lr = cpd_average_score  
  147.	  
  148.	            self.color_model = (1 - lr) * self.color_model + lr * obs_color_hist  
  149.	            self.motion_model += lr * motion_residual  # TODO: check is it right update method?  
  150.	  
  151.	            # set corrected present state  
  152.	            # corr_box = obs_tracking_confidence * obs_box + (1 - obs_tracking_confidence) * self.get_est_motion_box()  
  153.	            corr_box = 0.5 * obs_box + 0.5 * self.get_est_motion_box()  
  154.	            self.all_boxes.append(np.append(frame, corr_box))  
  155.	            self.all_particles.append(np.array(particle))  
  156.	            self.weights.append(weights)  
  157.	            self.frame = frame  
  158.	            self.tracking_confidence.append((self.get_last_tracking_confidence() + obs_tracking_confidence) / 2)  
  159.	  
  160.	  
  161.	  
  162.	        # if 'birth' state trajectory become confident  
  163.	        if self.get_last_tracking_confidence() > 0.7 and self.state == 'birth' and len(self.all_boxes) > 3:  
  164.	            self.state = 'track'  
  165.	            self.tracked = True  
  166.	        # if trajectory become low confident  
  167.	        elif self.get_last_tracking_confidence() < 0.5:  
  168.	            self.state = 'death'  
  169.	  
  170.	        return self.state
  ```

  line 172~: ‘re birth’- 물체가 death(물체가 t에는 존재하지 않지만 t-1에서는 존재할 때)인 상태일 때 새로 만들어진 물체(birth)의 위치(궤도)가 death인 것과 유사하면 re birth로 간주

  ```python
  172.	    def re_birth(self, frame, tid, particles, color_feats, lr=0.5):  
  173.	        """ 
  174.	        're-birth' state transaction. if 'birth' candidate has similar appearance with currently dead trajectory, 
  175.	        re-use it's identity since we consider those are same object. 
  176.	        Args: 
  177.	            frame: int, indicate frame number 
  178.	            tid: int, indicate tracking identity(ID) 
  179.	            particles: float numpy array, shape (n_particles, 5), number of particles to update 
  180.	                        each column means (x0, y0, x1, y1, weight) 
  181.	            color_feats: float numpy array, shape (n_particles, color_histogram_dimension) 
  182.	                        color histogram feature extracted from each particle 
  183.	 
  184.	        """  
  185.	        self.state = 'track'  # state {birth, track, death}  
  186.	        self.all_particles.append(particles[:, 0:5])  # [x1, y1, x2, y2, weights]  
  187.	        self.weights.append(particles[:, 4] / np.sum(particles[:, 4]))  
  188.	        box = weighted_sum(particles[:, 0:4], self.get_last_weights())  # point estimation by computing the marginal mean  
  189.	        self.all_boxes.append(np.append(frame, box))  # [frame_num, x1, y1, x2, y2, score]  
  190.	        self.frame = frame  # last frame's number  
  191.	  
  192.	        # appearance (color) model update  
  193.	        self.color_model = (1-lr) * self.color_model + lr * weighted_sum(color_feats, self.get_last_weights())  
  194.	  
  195.	        # initialize tracking confidence by weighted sum of detection confidence  
  196.	        self.tracking_confidence.append(np.sum(particles[:, 4] * self.get_last_weights().T))  
  197.	  
  198.	  
  199.	    def calc_motion_residual(self, obs_box):  
  200.	        """ 
  201.	        Calculate motion difference for update purpose which is called residual 
  202.	        Args: 
  203.	            obs_box: float numpy array, shape (1, 4), observed region box 
  204.	                    each column means (x0, y0, x1, y1) 
  205.	 
  206.	        Returns: 
  207.	            motion_diff: float numpy array, shape (1, 4) 
  208.	                        each column means (x velocity, y velocity, w velocity, h velocity) 
  209.	                        calculated motion difference between prediction and observation 
  210.	        """  
  211.	        # Motion (vx, vy, vw, vh)  
  212.	        cur_cx, cur_cy, cur_w, cur_h = xyxy_to_xywh(obs_box)  
  213.	        pred_cx, pred_cy, pred_w, pred_h = xyxy_to_xywh(self.get_est_motion_box())  
  214.	        diff_cx = cur_cx - pred_cx  
  215.	        diff_cy = cur_cy - pred_cy  
  216.	        diff_w = cur_w - pred_w  
  217.	        diff_h = cur_h - pred_h  
  218.	        motion_diff = [diff_cx, diff_cy, diff_w, diff_h]  
  219.	        return np.array(motion_diff)  
  220.	  
  221.	    def get_est_motion_particle(self):  
  222.	        """ 
  223.	        previous particles are moved by using motion model that can estimate current particle's position 
  224.	        """  
  225.	        # Motion (vx, vy, vw, vh)  
  226.	        last_particle = self.get_last_particle()  
  227.	        est_particle = []  
  228.	        for p in last_particle:  
  229.	            cx, cy, w, h = xyxy_to_xywh(p)  
  230.	            est_cx = cx + self.motion_model[0]  
  231.	            est_cy = cy + self.motion_model[1]  
  232.	            est_w = w + self.motion_model[2]  
  233.	            est_h = h + self.motion_model[3]  
  234.	            est_particle.append(np.append(xywh_to_xyxy([est_cx, est_cy, est_w, est_h]), p[4]))  
  235.	  
  236.	        return np.array(est_particle)  
  237.	  
  238.	    def get_est_motion_box(self):  
  239.	        """get current estimated posterior based on previous time step"""  
  240.	        est_particle = self.get_est_motion_particle()  
  241.	        weights = self.get_last_weights()  
  242.	        box = weighted_sum(est_particle[:, 0:4], weights)  
  243.	  
  244.	        return box  
  245.	  
  246.	    def get_last_tracking_confidence(self):  
  247.	        return self.tracking_confidence[-1]  
  248.	  
  249.	    def get_last_box(self):  
  250.	        return self.all_boxes[-1][1:5]  
  251.	  
  252.	    def get_last_particle(self):  
  253.	        return self.all_particles[-1][:]  
  254.	  
  255.	    def get_last_weights(self):  
  256.	        return self.weights[-1]  
  257.	  
  258.	    def get_tid(self):  
  259.	        return self.tid  
  260.	  
  261.	    def get_score_avr(self):  
  262.	        return np.sum(np.array(self.all_boxes)[:, 5]) / len(self.all_boxes)  
  263.	  
  264.	    def get_color_hist(self):  
  265.	        return self.color_model  
  266.	  
  267.	    def get_state(self):  
  268.	        return self.state  
  269.	  
  270.	    def get_frame(self):  
  271.	        return self.frame  
  ```



