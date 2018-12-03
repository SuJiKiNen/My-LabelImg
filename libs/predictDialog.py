
try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.lib import newIcon, labelValidator
from predict.AlexPredict import predict

BB = QDialogButtonBox


class PredictDialog(QDialog):

    def __init__(self, text="类别预测", parent=None):
        super(PredictDialog, self).__init__(parent)

        self.resize(500,300)
        self.center()
        self.setWindowTitle(text)
        self.state = QLabel()
        self.state.setText('处理中....')
        self.predict_result = QLabel()
        self.predict_first = QLabel()
        self.predict_second = QLabel()
        self.predict_third = QLabel()
        self.predict_fourth = QLabel()
        self.predict_fifth = QLabel()
        self.predicts = [self.predict_first,self.predict_second,self.predict_third,
                        self.predict_fourth,self.predict_fifth]

        layout = QVBoxLayout()
        layout.addWidget(self.state)
        layout.addWidget(self.predict_result)
        layout.addWidget(self.predict_first)
        layout.addWidget(self.predict_second)
        layout.addWidget(self.predict_third)
        layout.addWidget(self.predict_fourth)
        layout.addWidget(self.predict_fifth)

        self.buttonBox = bb = BB(BB.Ok, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(newIcon('done'))
        bb.accepted.connect(self.close)

        layout.addWidget(bb)
        self.setLayout(layout)

    def predictClass(self,image):
        self.state.setText('预测结果：')

        boolean,nameList,probList = predict(image)
        if not boolean:
            self.predict_result.setText('不支持的图片格式！channel数不为3！')
            return 

        self.predict_result.setText('该细胞类别是:' + nameList[0])

        for i in range(5):
            self.predicts[i].setText(nameList[i] + ' ' + str(probList[i]))


    def center(self):
        screen = QDesktopWidget().screenGeometry()       
        size = self.geometry()       
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)
