<annotation>
    <folder>VOC2012</folder>
    <filename>2007_000392.jpg</filename>                               //文件名
    <source>                                                           //图像来源（不重要）
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
    </source>
    <size>                                               //图像尺寸（长宽以及通道数）
        <width>500</width>
        <height>332</height>
        <depth>3</depth>
    </size>
    <segmented>1</segmented>                                   //是否用于分割（在图像物体识别中01无所谓）
    <object>                                                           //检测到的物体
        <name>horse</name>                                         //物体类别
        <pose>Right</pose>                                         //拍摄角度
        <truncated>0</truncated>                                   //是否被截断（0表示完整）
        <difficult>0</difficult>                                   //目标是否难以识别（0表示容易识别   ）
        <bndbox>                                                   //bounding-box（包含左下角和右上角xy坐标）
            <xmin>100</xmin>
            <ymin>96</ymin>
            <xmax>355</xmax>
            <ymax>324</ymax>
        </bndbox>
    </object>
    <object>                                                           //检测到多个物体
        <name>person</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>198</xmin>
            <ymin>58</ymin>
            <xmax>286</xmax>
            <ymax>197</ymax>
        </bndbox>
    </object>
</annotation>  
