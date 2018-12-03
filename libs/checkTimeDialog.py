try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.lib import newIcon, labelValidator

BB = QDialogButtonBox


class CheckTimeDialog(QDialog):

    def __init__(self, text="核验日志", parent=None):
        super(CheckTimeDialog, self).__init__(parent)

        self.resize(250,150)
        self.center()
        self.setWindowTitle(text)
        self.label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.buttonBox = bb = BB(BB.Ok, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(newIcon('done'))
        bb.accepted.connect(self.close)

        layout.addWidget(bb)
        self.setLayout(layout)

    def center(self):
        screen = QDesktopWidget().screenGeometry()       
        size = self.geometry()       
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)


    def displayCheck(self,text):
        self.label.setText(text)