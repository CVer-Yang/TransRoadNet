# TransRoadNet

This is a repository about the paper "TransRoadNet: A Novel Road Extraction Method for Remote Sensing Images via Combining High-level Semantic Feature and Context", accepted by IEEE GRSL 2022.
Road extraction is a significant research hotspot in the area of remote sensing images. Extracting an accurate road network from remote sensing images is still challeng-
ing because some objects in the images are similar to the road, and some results are discontinuous due to the occlusion. Recently, convolutional neural networks (CNNs) have shown their power in road extraction process. However, the contextual information can not be captured effectively by those CNNs. Based on CNNs, combining with high-level semantic features and foreground contextual information, a novel road extraction method for remote sensing images is proposed in this paper. Firstly, the position attention mechanism is designed to enhance the expression ability for the road feature. Then the contextual information extraction module (CIEM) is constructed to capture the road contextual information in the images. At last, foreground contextual information supplement module (FCISM) is proposed to provide foreground context information at different stages of the decoder, which can improve the inference ability for the occluded area. Extensive experiments on the DeepGlobal road dataset showed the proposed method outperforms the existing methods in accuracy, IoU, Precision, F1-score, and yields competitive recall results, which demonstrated the efficiency of the new model.

Note: Recenetly, we found a samll mistake of picture in TransRoadNet, and we have corrected it here. This slight difference does not impact the effectiveness of the experimental analysis and comparison.

Hope that the peers could understand and accommodate.
![image text]( https://github.com/CVer-Yang/TransRoadNet/blob/main/TransRoadNet.tif" This is the basic structure of the proposed TransRoadNet.")

If you have any questions about this paper, please send an e-mail to 793665968@qq.com.

