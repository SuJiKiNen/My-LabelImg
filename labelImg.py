#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import distutils.spawn
import os.path
import time
import platform
import re
import sys
import subprocess
import cv2
import numpy as np

from functools import partial
from collections import defaultdict
#from libtiff import TIFF

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    from PyQt5 import QtWidgets
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import resources
# Add internal libs
from libs.constants import *
from libs.lib import struct, newAction, newIcon, addActions, fmtShortcut, generateColorByText
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.checkTimeDialog import CheckTimeDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError
#from libs.predictDialog import PredictDialog
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.yolo_io import YoloReader
from libs.yolo_io import TXT_EXT
from libs.ustr import ustr
from libs.version import __version__

__appname__ = u'图片标注与切分软件'
DEFAULT_LAST_CHECK = "目前还没有核验过！"

# Utility functions and classes.

def have_qstring():
    '''p3/qt5 get rid of QString wrapper as py3 has native unicode str type'''
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))

def util_qt_strlistclass():
    return QStringList if have_qstring() else list

class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super(LoginDialog, self).__init__(parent)
        usr = QLabel("用户：")
        self.usrLineEdit = QLineEdit()
        self.user = None

        gridLayout = QGridLayout()
        gridLayout.addWidget(usr, 0, 0, 1, 1)
        gridLayout.addWidget(self.usrLineEdit, 0, 1, 1, 3)

        okBtn = QPushButton("确定")
        cancelBtn = QPushButton("取消")
        btnLayout = QHBoxLayout()

        btnLayout.setSpacing(60)
        btnLayout.addWidget(okBtn)
        btnLayout.addWidget(cancelBtn)

        dlgLayout = QVBoxLayout()
        #dlgLayout.setContentsMargins(40, 40, 40, 40)
        dlgLayout.addLayout(gridLayout)
        #dlgLayout.addStretch(40)
        dlgLayout.addLayout(btnLayout)

        self.setLayout(dlgLayout)
        okBtn.clicked.connect(self.accept)
        cancelBtn.clicked.connect(self.reject)
        self.setWindowTitle("登录")
        self.resize(400, 200)

    def accept(self):
        if not self.usrLineEdit.text().strip():
            QMessageBox.warning(self,
                    "警告",
                    "用户名为空！",
                    QMessageBox.Yes)
            self.usrLineEdit.setFocus()
        else:
            self.user = self.usrLineEdit.text().strip()
            main_win = MainWindow(userName=self.user)
            main_win.show()
            super(LoginDialog, self).accept()

    def reject(self):
        QMessageBox.warning(self,
                            "退出",
                            "确定退出？",
                            QMessageBox.Yes)
        sys.exit()


class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


# PyQt5: TypeError: unhashable type: 'QListWidgetItem'
class HashableQListWidgetItem(QListWidgetItem):

    def __init__(self, *args):
        super(HashableQListWidgetItem, self).__init__(*args)

    def __hash__(self):
        return hash(id(self))


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self,defaultFilename=None, defaultPrefdefClassFile=None, defaultSaveDir=None, defaultPicSaveDir=None,userName = None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        # Save divided pictures
        self.defaultPicSaveDir = defaultPicSaveDir

        # Current userName
        self.userName = userName

        # Operation user
        self.opUserName = None

        # Operation time
        self.opTime = None

        # Save as Pascal voc xml
        self.defaultSaveDir = defaultSaveDir
        self.usingPascalVocFormat = True
        self.usingYoloFormat = False

        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None

        # Whether we need to save or not.
        self.dirty = False
        self.saveTimes = 0
        self.isChecked = False
        self.lastCheckTime = DEFAULT_LAST_CHECK

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = self.getAvailableScreencastViewer()
        #self.screencast = "https://youtu.be/p0nR2YsCY_U"
        self.screencast = "http://120.79.231.160/labelImg"

        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)
        self.checkTimeDialog = CheckTimeDialog(parent=self)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        userInfoLayout = QGridLayout()
        userInfoLayout.setContentsMargins(0, 0, 0, 0)
        self.curUserLabel = QLabel("\n  当前用户: ")
        self.curUserNameText = QLabel("\n" + self.userName)
        self.opUserLabel = QLabel("  标注者：")
        self.opUserNameText = QLabel(self.opUserName)
        self.opTimeLabel = QLabel("  标注时间:\n")
        self.opTimeText = QLabel(self.opTime)
        self.redirectBox = QCheckBox(u'我要核验')
        self.redirectBox.setChecked(False)

        userInfoLayout.setSpacing(15)
        userInfoLayout.addWidget(self.curUserLabel,1,0,1,1)
        userInfoLayout.addWidget(self.curUserNameText,1,1,1,1)
        userInfoLayout.addWidget(self.redirectBox,1,2,1,1)
        userInfoLayout.addWidget(self.opUserLabel,2,0,1,1)
        userInfoLayout.addWidget(self.opUserNameText,2,1,1,1)
        userInfoLayout.addWidget(self.opTimeLabel,3,0,1,1)
        userInfoLayout.addWidget(self.opTimeText,3,1,1,1)
        userInfoContainer = QWidget()
        userInfoContainer.setLayout(userInfoLayout)

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        # Create a widget for using default label
        self.useDefaultLabelCheckbox = QCheckBox(u'使用默认标签')
        self.useDefaultLabelCheckbox.setChecked(False)
        self.defaultLabelTextLine = QLineEdit()
        useDefaultLabelQHBoxLayout = QHBoxLayout()
        useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        useDefaultLabelContainer = QWidget()
        useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)

        # Create a widget for edit and diffc button
        self.diffcButton = QCheckBox(u'difficult')
        self.diffcButton.setChecked(False)
        self.diffcButton.stateChanged.connect(self.btnstate)
        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to listLayout
        listLayout.addWidget(userInfoContainer)
        listLayout.addWidget(self.editButton)
        listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(useDefaultLabelContainer)

        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)

        self.dock = QDockWidget(u'用户与标签', self)
        self.dock.setObjectName(u'Labels')
        self.dock.setWidget(labelListContainer)

        # Tzutalin 20160906 : Add file list and dock to move faster
        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        #self.checkListWidget = QListWidget() #+
        filelistLayout = QHBoxLayout()  #gai V->H
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        #filelistLayout.addWidget(self.checkListWidget,stretch=1)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(u'文件列表', self)
        self.filedock.setObjectName(u'Files')
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        # Tzutalin 20160906 : Add file list and dock to move faster
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.filedock.setFeatures(QDockWidget.DockWidgetFloatable)

        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

        # Actions
        action = partial(newAction, self)
        quit = action('&退出', self.close,
                      'Ctrl+Q', 'quit', u'退出应用')

        PredefinedClass = action('&打开标注目录', self.importClass,
                                 'Ctrl+M', 'open', u'改变默认的标注目录')

        open = action('&打开', self.openFile,
                      'Ctrl+O', 'open', u'打开图片与标签')

        opendir = action('&打开文件夹', self.openDirDialog,
                         'Ctrl+u', 'open', u'打开文件夹')

        changeSavedir = action('&改变标注位置', self.changeSavedirDialog,
                               'Ctrl+r', 'open', u'改变标注文件打开与保存的文件夹位置')

        pictureSavedir = action('&图片存储位置', self.pictureSavedirDialog,
                                'Ctrl+P', 'open', u'改变切分后图片存储的文件夹位置')

        openAnnotation = action('&打开标注', self.openAnnotationDialog,
                                'Ctrl+Shift+O', 'open', u'打开标注文件')

        openNextImg = action('&下一张', self.openNextImg,
                             'd', 'next', u'打开下一张')

        openPrevImg = action('&上一张', self.openPrevImg,
                             'a', 'prev', u'打开上一张')

        verify = action('&高亮', self.verifyImg,
                        'space', 'verify', u'高亮图片')

        save = action('&保存', self.saveFile,
                      'Ctrl+S', 'save', u'将标注保存至文件', enabled=False)

        savePicture = action('&保存切分图片', self.saveDividedPicture,
                      'Ctrl+Shift+P', 'save', u'将切分后的图片保存至选定文件夹', enabled=True)

        save_format = action('&PascalVOC', self.change_format,
                      'Ctrl+', 'format_voc', u'改变存储方式', enabled=True)

        saveAs = action('&另存为', self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', u'将标签保存至其他文件夹', enabled=False)

        close = action('&关闭', self.closeFile, 'Ctrl+W', 'close', u'关闭当前文件')

        resetAll = action('&全部重设', self.resetAll, None, 'resetall', u'全部重设')

        color1 = action('标注框颜色', self.chooseColor1,
                        'Ctrl+L', 'color_line', u'选择标注框线条的颜色')

        createMode = action('创建标注框', self.setCreateMode,
                            'w', 'new', u'开始绘制一个矩形标注框', enabled=False)
        editMode = action('&编辑标注框', self.setEditMode,
                          'Ctrl+J', 'edit', u'移动与编辑标注框', enabled=False)

        create = action('创建标注框', self.createShape,
                        'w', 'new', u'开始绘制一个矩形标注框', enabled=False)
        delete = action('删除标注框', self.deleteSelectedShape,
                        'Delete', 'delete', u'删除', enabled=False)
        copy = action('&复制标注框', self.copySelectedShape,
                      'Ctrl+D', 'copy', u'创建一个与之相同的标注框',
                      enabled=False)

        advancedMode = action('&单框修改模式', self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', u'返回单框修改模式',
                              checkable=True)

        hideAll = action('&隐藏标注框', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', u'隐藏所有矩形',
                         enabled=False)
        showAll = action('&显示标注框', partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', u'显示所有矩形',
                         enabled=False)

        help = action('&教学', self.showTutorialDialog, None, 'help', u'展示教学视频')
        showInfo = action('&信息', self.showInfoDialog, None, 'help', u'详细信息')

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action('放 &大', partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', u'增加缩放等级', enabled=False)
        zoomOut = action('&缩 小', partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', u'减小缩放等级', enabled=False)
        zoomOrg = action('&还原 大小', partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', u'还原至原大小', enabled=False)
        fitWindow = action('&适应 窗口', self.setFitWindow,
                           'Ctrl+F', 'fit-window', u'随窗口大小缩放',
                           checkable=True, enabled=False)
        fitWidth = action('适应 &宽度', self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', u'随窗口宽度缩放',
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action('&编辑标签', self.editLabel,
                      'Ctrl+E', 'edit', u'编辑选中矩形标注框的标签',
                      enabled=False)
        self.editButton.setDefaultAction(edit)

        shapeLineColor = action('矩形 &线条颜色', self.chshapeLineColor,
                                icon='color_line', tip=u'Change the line color for this specific shape',
                                enabled=False)
        shapeFillColor = action('矩形 &填充颜色', self.chshapeFillColor,
                                icon='color', tip=u'Change the fill color for this specific shape',
                                enabled=False)
        #predictClass = action('预测细胞类型',self.predictCellClass,tip=u'用深度学习模型预测选中矩形框内的细胞类型',enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText('显示/隐藏 标签栏')
        labels.setShortcut('Ctrl+Shift+L')

        # Lavel list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = struct(save=save, save_format=save_format, saveAs=saveAs, savePicture=savePicture, open=open, close=close, resetAll = resetAll,
                              lineColor=color1, create=create, delete=delete, edit=edit, copy=copy,
                              createMode=createMode, editMode=editMode, advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,#predictClass=predictClass,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              fileMenuActions=(
                                  open, opendir, save, saveAs, savePicture, close, resetAll, quit, PredefinedClass),
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete,
                                        None, color1),
                              beginnerContext=(create, edit, copy, delete),
                              advancedContext=(createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),#,predictClass),
                              onLoadActive=(
                                  close, create, createMode, editMode),
                              onShapesPresent=(saveAs, hideAll, showAll))

        self.menus = struct(
            file=self.menu('&文件'),
            edit=self.menu('&编辑'),
            view=self.menu('&视图'),
            check=self.menu('核验'),
            help=self.menu('&帮助'),
            recentFiles=QMenu('打开 &最近文件'),
            labelList=labelMenu)

        # Auto saving : Enable auto saving if pressing next
        self.autoSaving = QAction("自动保存", self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(settings.get(SETTING_AUTO_SAVE, False))
        # Sync single class mode from PR#106
        self.singleClassMode = QAction("单类模式", self)
        self.singleClassMode.setShortcut("Ctrl+Shift+S")
        self.singleClassMode.setCheckable(True)
        self.singleClassMode.setChecked(settings.get(SETTING_SINGLE_CLASS, False))
        self.lastLabel = None
        # Add option to enable/disable labels being painted at the top of bounding boxes
        self.paintLabelsOption = QAction("显示标签", self)
        self.paintLabelsOption.setShortcut("Ctrl+Shift+P")
        self.paintLabelsOption.setCheckable(True)
        self.paintLabelsOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.paintLabelsOption.triggered.connect(self.togglePaintLabelsOption)
        # Add the re-check function
        self.check = QAction("核验",self)
        self.check.setShortcut("Ctrl+C")
        self.check.setEnabled(True)
        self.check.triggered.connect(self.checkCorrect)

        # Display whether the pixmap has been checked
        self.isCheck = QAction("已核验",self)
        self.isCheck.setCheckable(True)
        self.isCheck.setEnabled(False)
        self.isCheck.setChecked(False)
        self.isCheck.triggered.connect(self.keepIsCheck)

        # Cancel the check
        self.cancelCheck = QAction("取消核验",self)
        self.cancelCheck.setCheckable(False)
        self.cancelCheck.setEnabled(True)
        self.cancelCheck.triggered.connect(self.checkCancel)

        #核验日志
        self.checkLog = QAction("核验日志",self)
        self.checkLog.triggered.connect(self.popCheckInfo)

        addActions(self.menus.file,
                   (open, opendir, PredefinedClass, changeSavedir, openAnnotation, self.menus.recentFiles, save, save_format, saveAs, pictureSavedir, savePicture, close, resetAll, quit))
        addActions(self.menus.help, (help, showInfo))
        addActions(self.menus.view, (
            self.autoSaving,
            self.singleClassMode,
            self.paintLabelsOption,
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))
        
        addActions(self.menus.check,(self.check,self.isCheck,self.cancelCheck,self.checkLog))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&复制到当前位置', self.copyShape),
            action('&移动到当前位置', self.moveShape)))

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            open, opendir, PredefinedClass, changeSavedir, pictureSavedir, openNextImg, openPrevImg, verify, save, savePicture, None,create, copy, delete, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self.actions.advanced = (
            open, opendir, PredefinedClass, changeSavedir, pictureSavedir, openNextImg, openPrevImg, save, savePicture, save_format, None,
            createMode, editMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(defaultFilename)
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = settings.get(SETTING_WIN_POSE, QPoint(0, 0))
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        if self.defaultSaveDir is None and saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.defaultSaveDir))
            self.statusBar().show()

        picSaveDir = ustr(settings.get(SETTING_PIC_SAVE_DIR, None))
        if self.defaultPicSaveDir is None and picSaveDir is not None and os.path.exists(picSaveDir):
            self.defaultPicSaveDir = picSaveDir

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if deafult file
        if self.filePath and os.path.isdir(self.filePath):
            self.openDirDialog(dirpath=self.filePath)

    ## Support Functions ##
    def set_format(self, save_format):
        if save_format == FORMAT_PASCALVOC:
            self.actions.save_format.setText(FORMAT_PASCALVOC)
            self.actions.save_format.setIcon(newIcon("format_voc"))
            self.usingPascalVocFormat = True
            self.usingYoloFormat = False
            LabelFile.suffix = XML_EXT

        elif save_format == FORMAT_YOLO:
            self.actions.save_format.setText(FORMAT_YOLO)
            self.actions.save_format.setIcon(newIcon("format_yolo"))
            self.usingPascalVocFormat = False
            self.usingYoloFormat = True
            LabelFile.suffix = TXT_EXT

    def change_format(self):
        if self.usingPascalVocFormat: self.set_format(FORMAT_YOLO)
        elif self.usingYoloFormat: self.set_format(FORMAT_PASCALVOC)

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            #self.actions.predictClass.setEnabled(True)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner()\
            else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def getAvailableScreencastViewer(self):
        osName = platform.system()

        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open', '-a', 'Safari']

    ## Callbacks ##
    def showTutorialDialog(self):
        subprocess.Popen(self.screencastViewer + [self.screencast])

    def showInfoDialog(self):
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)
        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(QColor(255, 255, 255, 100))
            self.setDirty()

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        currIndex = self.mImgList.index(ustr(item.text().split()[1]))
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.loadFile(filename)

    # Add chris
    def btnstate(self, item= None):
        """ Function to handle difficult examples
        Update on each object """
        if not self.canvas.editing():
            return

        item = self.currentItem()
        if not item: # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count()-1)

        difficult = self.diffcButton.isChecked()

        try:
            shape = self.itemsToShapes[item]
        except:
            pass
        # Checked and Update
        try:
            if difficult != shape.difficult:
                shape.difficult = difficult
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape):
        shape.paintLabel = self.paintLabelsOption.isChecked()
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(QColor(255, 255, 255, 100))
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        self.labelList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult in shapes:
            shape = Shape(label=label)
            for x, y in points:
                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.close()
            s.append(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = generateColorByText(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generateColorByText(label)

            self.addLabel(shape)

        self.canvas.loadShapes(s)

    def saveLabels(self, annotationFilePath):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                       # add chris
                        difficult = s.difficult)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add differrent annotation formats here
        self.saveTimes += 1
        if not self.redirectBox.isChecked():
            self.opUserName = self.userName
            self.opTime = time.ctime()
        self.updateOpInfo()
        try:
            if self.usingPascalVocFormat is True:
                if ustr(annotationFilePath[-4:]) != ".xml":
                    annotationFilePath += XML_EXT
                print ('Img: ' + self.filePath + ' -> Its xml: ' + annotationFilePath)
                
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes, self.filePath, self.imageData, self.saveTimes,self.opUserName,
                                                   self.opTime,self.isChecked,self.lastCheckTime,self.lineColor.getRgb(), self.fillColor.getRgb())
            elif self.usingYoloFormat is True:
                if annotationFilePath[-4:] != ".txt":
                    annotationFilePath += TXT_EXT
                print ('Img: ' + self.filePath + ' -> Its txt: ' + annotationFilePath)
                self.labelFile.saveYoloFormat(annotationFilePath, shapes, self.filePath, self.imageData, self.labelHist,
                                                   self.isChecked,self.lastCheckTime,self.lineColor.getRgb(), self.fillColor.getRgb())
            else:
                self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
                                    self.lineColor.getRgb(), self.fillColor.getRgb())
            return True
        except LabelFileError as e:
            self.saveTimes -= 1
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]
            # Add Chris
            self.diffcButton.setChecked(shape.difficult)

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if not self.useDefaultLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            # Sync single class mode from PR#106
            if self.singleClassMode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.labelDialog.popUp(text=self.prevLabelText)
                self.lastLabel = text
        else:
            text = self.defaultLabelTextLine.text()

        # Add Chris
        self.diffcButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            generate_color = self.lineColor
            shape = self.canvas.setLastLabel(text, generate_color, generate_color)
            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def checkCorrect(self):
        if self.filePath == None:
            self.errorMessage(u'错误!',u'请先打开图片！')
            return              

        reply = QMessageBox.information(self, '确认', '本图片将被标注为已核验，请保存以生效',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.toggleIsChecked(True)
            self.lastCheckTime = time.asctime( time.localtime(time.time()))
            self.setDirty()
            self.saveFile()
            self.sortFileList() #+
        else:
            pass

    #do nothing if the ischecked button is pressed as
    #it is just used for displaying

    def toggleIsChecked(self,value):
        self.isChecked = value
        self.isCheck.setEnabled(value)
        self.isCheck.setChecked(value)

    def keepIsCheck(self):
        #prevent the press op from changing the isChecked value
        self.isCheck.setChecked(self.isChecked)

    def popCheckInfo(self):
        if self.lastCheckTime == DEFAULT_LAST_CHECK:
            self.checkTimeDialog.displayCheck(self.lastCheckTime)
        else:
            self.checkTimeDialog.displayCheck("上次核验时间为：\n" + self.lastCheckTime)
        self.checkTimeDialog.show()

    def checkCancel(self):
        self.toggleIsChecked(False)
        self.setDirty()
        self.saveFile()   
        self.sortFileList()   

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)

        unicodeFilePath = ustr(filePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            index = self.mImgList.index(unicodeFilePath)
            fileWidgetItem = self.fileListWidget.item(index)
            fileWidgetItem.setSelected(True)

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                except LabelFileError as e:
                    self.errorMessage(u'Error opening file',
                                      (u"<p><b>%s</b></p>"
                                       u"<p>Make sure <i>%s</i> is a valid label file.")
                                      % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)
                    return False
                self.imageData = self.labelFile.imageData
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
                self.canvas.verified = self.labelFile.verified
            else:
                # Load image:
                # read data first and store for saving into label file.
                if filePath.split('.')[-1] == 'tif':
                    ###tif = TIFF.open(unicodeFilePath, mode='r')
                    ###self.imageData = tif.read_image()
                    self.imageData = cv2.imdecode(np.fromfile(unicodeFilePath,dtype=np.uint8),-1)
                    ##image = CV2QImage(self.imageData)
                    Img = self.imageData
                    height, width, bytesPerComponent = Img.shape
                    bytesPerLine = 3 * width
                    cv2.cvtColor(Img, cv2.COLOR_BGR2RGB, Img)
                    image = QImage(Img.data, width, height, bytesPerLine,QImage.Format_RGB888)
                else:
                    self.imageData = read(unicodeFilePath, None)
                    image = QImage.fromData(self.imageData)
                self.labelFile = None
                self.canvas.verified = False

            #image = QImage.fromData(self.imageData)
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            # Label xml file and show bound box according to its filename
            # if self.usingPascalVocFormat is True:
            if self.defaultSaveDir is not None:
                basename = os.path.basename(
                    os.path.splitext(self.filePath)[0])
                xmlPath = os.path.join(self.defaultSaveDir, basename + XML_EXT)
                txtPath = os.path.join(self.defaultSaveDir, basename + TXT_EXT)

                """Annotation file priority:
                PascalXML > YOLO
                """
                if os.path.isfile(xmlPath):
                    self.loadPascalXMLByFilename(xmlPath)
                elif os.path.isfile(txtPath):
                    self.loadYOLOTXTByFilename(txtPath)
                else:
                    self.saveTimes = 0
                    self.isChecked=False
                    self.lastCheckTime = DEFAULT_LAST_CHECK
                    self.toggleIsChecked(self.isChecked)
                    self.opUserName = "无"
                    self.opTime = "无"
            else:
                xmlPath = os.path.splitext(filePath)[0] + XML_EXT
                txtPath = os.path.splitext(filePath)[0] + TXT_EXT
                if os.path.isfile(xmlPath):
                    self.loadPascalXMLByFilename(xmlPath)
                elif os.path.isfile(txtPath):
                    self.loadYOLOTXTByFilename(txtPath)
                else:
                    self.saveTimes = 0
                    self.isChecked=False
                    self.lastCheckTime = DEFAULT_LAST_CHECK
                    self.toggleIsChecked(self.isChecked)
                    self.opUserName = "无"
                    self.opTime = "无"

            self.updateOpInfo()
            self.sortFileList()
            
            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count()-1))
                self.labelList.item(self.labelList.count()-1).setSelected(True)

            self.canvas.setFocus(True)
            return True
        return False
    
    def updateOpInfo(self):
        if not self.redirectBox.isChecked():
            self.opUserNameText.setText(self.opUserName)
            self.opTimeText.setText(self.opTime)

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        settings = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ""

        if self.defaultPicSaveDir and os.path.exists(self.defaultPicSaveDir):
            settings[SETTING_PIC_SAVE_DIR] = ustr(self.defaultPicSaveDir)
        else:
            settings[SETTING_PIC_SAVE_DIR] = ""

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ""

        settings[SETTING_AUTO_SAVE] = self.autoSaving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.paintLabelsOption.isChecked()
        settings.save()
    ## User Dialogs ##

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        extensions.append('.tif')
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images.append(path)
        images.sort(key=lambda x: x.lower())
        return images

    def changeSavedirDialog(self, _value=False):
        if self.defaultSaveDir is not None:
            path = ustr(self.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                       '%s - Save annotations to the directory' % __appname__, path,  QFileDialog.ShowDirsOnly
                                                       | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSaveDir = dirpath

        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

    def pictureSavedirDialog(self, _value = False):
        if self.defaultPicSaveDir is not None:
            path = ustr(self.defaultPicSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                       '%s - Save divided pictures to the directory' % __appname__, path,  QFileDialog.ShowDirsOnly
                                                       | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultPicSaveDir = dirpath

        self.statusBar().showMessage('%s . Divided pictures will be saved to %s' %
                                     ('Change saved folder', self.defaultPicSaveDir))
        self.statusBar().show()

    def openAnnotationDialog(self, _value=False):
        if self.filePath is None:
            self.statusBar().showMessage('Please select image first')
            self.statusBar().show()
            return

        path = os.path.dirname(ustr(self.filePath))\
            if self.filePath else '.'
        if self.usingPascalVocFormat:
            filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
            filename = ustr(QFileDialog.getOpenFileName(self,'%s - Choose a xml file' % __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.loadPascalXMLByFilename(filename)

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'

        targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                     '%s - Open Directory' % __appname__, defaultOpenDirPath,
                                                     QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        self.importDirImages(targetDirPath)

    def importDirImages(self, dirpath):
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.dirname = dirpath
        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.sortFileList()
        self.openNextImg()
        """
        itemList = []
        for imgPath in self.mImgList:
            #item = QListWidgetItem(imgPath)
            #
            purePath = os.path.basename(imgPath).split('.')[0] + '.xml'
            if self.defaultSaveDir is not None and os.path.exists(os.path.join(self.defaultSaveDir, purePath)):
                item.setBackground(QColor(0, 255, 0, 100))
                itemList.append(item)
            else:
                self.fileListWidget.addItem(item)
        for item in itemList:
            self.fileListWidget.addItem(item)
            #
            self.fileListWidget.addItem(item)
        self.sortFileList()    #+
        """

    def verifyImg(self, _value=False):
        # Proceding next image without dialog if having any label
         if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.saveFile()
                if self.labelFile != None:
                    self.labelFile.toggleVerify()
                else:
                    return

            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.saveFile()

    def openPrevImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            if filename:
                self.loadFile(filename)

    def openNextImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]

        if filename:
            self.loadFile(filename)

    def importClass(self,_value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        filename = QFileDialog.getOpenFileName(self,"open predefined class",path,"Txt files(*.txt)")
        #"open file Dialog "为文件对话框的标题，第三个是打开的默认路径，第四个是文件类型过滤器
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            # Load predefined classes to the list
            self.loadPredefinedClasses(filename)
            self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix] + ['*%s' % '.tif'])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def saveFile(self, _value=False):
        if self.defaultSaveDir is not None and len(ustr(self.defaultSaveDir)):
            if self.filePath:
                imgFileName = os.path.basename(self.filePath)
                savedFileName = os.path.splitext(imgFileName)[0]
                savedPath = os.path.join(ustr(self.defaultSaveDir), savedFileName)
                self._saveFile(savedPath)
        else:
            imgFileDir = os.path.dirname(self.filePath)
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(imgFileName)[0]
            savedPath = os.path.join(imgFileDir, savedFileName)
            self._saveFile(savedPath if self.labelFile
                           else self.saveFileDialog())
        self.sortFileList()
        self.redirectFile()

    """
    def predictCellClass(self):

        if not self.canvas.selectedShape:
            self.errorMessage(u'错误',u'未选择标注框！')
            return
        predictShape = self.canvas.selectedShape
        shapeIndex = self.canvas.shapes.index(predictShape)
        shapeLabel = predictShape.label
        tmpIndex = 1
        tmpPath = os.path.join(os.path.join(self.defaultPicSaveDir,shapeLabel), 
                shapeLabel+'_'+os.path.basename(self.filePath).split('.')[0]+'_'+str(tmpIndex)+'.tif')
        
        while(not os.path.exists(tmpPath)):
            tmpIndex += 1
            if tmpIndex > 10000:
                self.errorMessage(u'错误',u'没有找到切分图片！请确认是否已保存！')
                return
        
        shapePath=tmpPath
        Img = cv2.imdecode(np.fromfile(shapePath,dtype=np.uint8),-1)
        dialog = PredictDialog(parent=self)
        dialog.predictClass(Img)
        dialog.show()
    """
    

    def saveDividedPicture(self, _value = False):
        if self.defaultPicSaveDir == None:
            self.errorMessage(u'Error saving pictures',
                                  u"<p>切分图片保存的文件夹尚未被选择.")
            return

        if self.filePath == None:
            self.errorMessage(u'Error saving pictures',
                                  u"<p>没有图片被打开.")
            return

        """
        for root, dirs, files in os.walk(self.defaultPicSaveDir):
            for dir in dirs:
                for root2, dirs2, files2 in os.walk(os.path.join(root,dir)):
                    for file in files2:
                        print(file.split('_')[1])
                        print(os.path.basename(self.filePath).split('.')[0])
                        if file.split('_')[1] == os.path.basename(self.filePath).split('.')[0]:
                            os.remove(os.path.join(root2,file))
                            print('Removed!')
        """
        # Replaced By The Code Below
        for sub_dir in os.listdir(self.defaultPicSaveDir):
            for cell in os.listdir(os.path.join(self.defaultPicSaveDir,sub_dir)):
                if cell.split('_')[1] == os.path.basename(self.filePath).split('.')[0]:
                    os.remove(os.path.join(os.path.join(self.defaultPicSaveDir,sub_dir),cell))

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                       # add chris
                        difficult = s.difficult)

        def get_picture_path(d, p, l):
            index = 1
            while (True):
                pp = os.path.join(d, l+'_'+p+'_'+str(index)+'.tif')
                if not os.path.exists(pp):
                    return pp
                else:
                    index = index + 1

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        for shape in shapes:
            points = shape['points']
            label = shape['label']
            difficult = int(shape['difficult'])
            bndbox = LabelFile.convertPoints2BndBox(points)

            subdirPath = os.path.join(self.defaultPicSaveDir,label)
            if not os.path.exists(subdirPath):
                os.mkdir(subdirPath)
            #print (get_picture_path(subdirPath, os.path.basename(self.filePath).split('.')[0], label))
            dividedPicturePath = get_picture_path(subdirPath, os.path.basename(self.filePath).split('.')[0], label)
            unicodeFilePath = ustr(self.filePath)
            Img = cv2.imdecode(np.fromfile(unicodeFilePath,dtype=np.uint8),-1)
            img = Img[bndbox[1]:bndbox[3],bndbox[0]:bndbox[2]]
            cv2.imencode('.tif',img)[1].tofile(dividedPicturePath)
        self.statusBar().showMessage('%s successfully divided and saved to %s' %
                                    (self.filePath, self.defaultPicSaveDir))
        self.statusBar().show()

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self.filePath)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            fullFilePath = ustr(dlg.selectedFiles()[0])
            return os.path.splitext(fullFilePath)[0] # Return file path without the extension.
        return ''

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):
        return not (self.dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'You have unsaved changes, proceed anyway?'
        return yes == QMessageBox.warning(self, u'Attention', msg, yes | no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if predefClassesFile != None:
            if os.path.exists(predefClassesFile) is True:
                self.labelHist = None
                with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                    for line in f:
                        line = line.strip()
                        if self.labelHist is None:
                            self.labelHist = [line]
                        else:
                            self.labelHist.append(line)

    def loadPascalXMLByFilename(self, xmlPath):
        if self.filePath is None:
            self.saveTimes = 0
            self.isChecked=False
            self.toggleIsChecked(self.isChecked)
            self.opUserName = "无"
            self.opTime = "无"
            return
        if os.path.isfile(xmlPath) is False:
            self.saveTimes = 0
            self.isChecked=False
            self.opUserName = "无"
            self.opTime = "无"
            self.toggleIsChecked(self.isChecked)
            return

        self.set_format(FORMAT_PASCALVOC)

        tVocParseReader = PascalVocReader(xmlPath)
        shapes = tVocParseReader.getShapes()
        self.saveTimes = tVocParseReader.getSaveTimes()
        self.isChecked = tVocParseReader.getChecked()
        self.lastCheckTime = tVocParseReader.getLastCheckTime()
        self.opUserName = tVocParseReader.getOpUserName()
        self.opTime = tVocParseReader.getOpTime()
        self.toggleIsChecked(self.isChecked)

        self.loadLabels(shapes)
        self.canvas.verified = tVocParseReader.verified

    def loadYOLOTXTByFilename(self, txtPath):
        if self.filePath is None:
            return
        if os.path.isfile(txtPath) is False:
            return

        self.set_format(FORMAT_YOLO)
        tYoloParseReader = YoloReader(txtPath, self.image)
        shapes = tYoloParseReader.getShapes()
        print (shapes)
        self.loadLabels(shapes)
        self.canvas.verified = tYoloParseReader.verified

    def togglePaintLabelsOption(self):
        paintLabelsOptionChecked = self.paintLabelsOption.isChecked()
        for shape in self.canvas.shapes:
            shape.paintLabel = paintLabelsOptionChecked

    def sortFileList(self):
        state_1,state_2,state_3 = [],[],[]

        for imgPath in self.mImgList:
            unicodeImgPath = ustr(imgPath)
            if self.defaultSaveDir is not None:
                basename = os.path.basename(
                    os.path.splitext(unicodeImgPath)[0])
                imgXmlPath = os.path.join(self.defaultSaveDir, basename + XML_EXT)
                imgTxtPath = os.path.join(self.defaultSaveDir, basename + TXT_EXT)
            else:
                imgXmlPath = os.path.splitext(unicodeImgPath)[0] + XML_EXT
                imgTxtPath = os.path.splitext(unicodeImgPath)[0] + TXT_EXT
            
            if not os.path.isfile(imgXmlPath):
                # not labelled 
                state_1.append(imgPath)
            else:
                tVocParseReader = PascalVocReader(imgXmlPath)
                imgChecked = tVocParseReader.getChecked()
                if not imgChecked:
                    # labelled but not checked
                    state_2.append(imgPath)
                else:
                    # checked
                    state_3.append(imgPath)
            self.mImgList = state_2 + state_1 + state_3
 
        #self.fileListWidget.clear()
        #for imgPath in self.mImgList:
            #item = QListWidgetItem(imgPath)
            #self.fileListWidget.addItem(item)
        self.loadCheckList()

        # Add the background color to highlight
        if self.filePath != None:
            index = self.mImgList.index(self.filePath)
            fileWidgetItem = self.fileListWidget.item(index)
            #fileWidgetItem.setSelected(True)
            fileWidgetItem.setBackground(QColor(0, 255, 0, 100))
        
    def redirectFile(self):
        state_1,state_2,state_3 = [],[],[]

        for imgPath in self.mImgList:
            unicodeImgPath = ustr(imgPath)
            if self.defaultSaveDir is not None:
                basename = os.path.basename(
                    os.path.splitext(unicodeImgPath)[0])
                imgXmlPath = os.path.join(self.defaultSaveDir, basename + XML_EXT)
                imgTxtPath = os.path.join(self.defaultSaveDir, basename + TXT_EXT)
            else:
                imgXmlPath = os.path.splitext(unicodeImgPath)[0] + XML_EXT
                imgTxtPath = os.path.splitext(unicodeImgPath)[0] + TXT_EXT
            
            if not os.path.isfile(imgXmlPath):
                # not labelled 
                state_1.append(imgPath)
            else:
                tVocParseReader = PascalVocReader(imgXmlPath)
                imgChecked = tVocParseReader.getChecked()
                if not imgChecked:
                    # labelled but not checked
                    state_2.append(imgPath)
                else:
                    # checked
                    state_3.append(imgPath)

        if self.redirectBox.isChecked():
            if len(state_2) > 0:
                self.filePath = state_2[0]
                self.loadFile(self.filePath)
        else:
            if len(state_1) > 0:
                self.filePath = state_1[0]
                self.loadFile(self.filePath)

    def loadCheckList(self):
        #self.checkListWidget.clear()
        self.fileListWidget.clear()
        for imgPath in self.mImgList:
            unicodeImgPath = ustr(imgPath)
            if self.defaultSaveDir is not None:
                basename = os.path.basename(
                    os.path.splitext(unicodeImgPath)[0])
                imgXmlPath = os.path.join(self.defaultSaveDir, basename + XML_EXT)
                imgTxtPath = os.path.join(self.defaultSaveDir, basename + TXT_EXT)
            else:
                imgXmlPath = os.path.splitext(unicodeImgPath)[0] + XML_EXT
                imgTxtPath = os.path.splitext(unicodeImgPath)[0] + TXT_EXT
            
            if not os.path.isfile(imgXmlPath):
                imgState = 0
            else:
                tVocParseReader = PascalVocReader(imgXmlPath)
                imgChecked = tVocParseReader.getChecked()
                if not imgChecked:
                    imgState = 1
                else:
                    imgState = 2
            stateList = ['未标注','未核验','已核验']
            #imgItem = QListWidgetItem(stateList[imgState])
            #self.checkListWidget.addItem(imgItem)
            self.fileListWidget.addItem(QListWidgetItem(str(stateList[imgState] + " " +  imgPath)))
            


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    # Usage : labelImg.py image predefClassFile saveDir
    win = MainWindow(argv[1] if len(argv) >= 2 else None,
                     argv[2] if len(argv) >= 3 else os.path.join(
                         os.path.dirname(sys.argv[0]),
                         'data', 'predefined_classes.txt'),
                     argv[3] if len(argv) >= 4 else None,
                     argv[4] if len(argv) >= 5 else None)
    win.show()
    return app, win


def CV2QImage(cv_image):
    width = cv_image.shape[1] #获取图片宽度
    height = cv_image.shape[0]  #获取图片高度
    pixmap = QPixmap(width, height) #根据已知的高度和宽度新建一个空的QPixmap,
    qimg = pixmap.toImage()  #将pximap转换为QImage类型的qimg
    #循环读取cv_image的每个像素的r,g,b值，构成qRgb对象，再设置为qimg内指定位置的像素
    for row in range(0, height):
        for col in range(0,width):
            r = cv_image[row,col,0]
            g = cv_image[row,col,1]
            b = cv_image[row,col,2]
            pix = qRgb(r, g, b)
            qimg.setPixel(col, row, pix)
    return qimg #转换完成，返回


def main():
    '''construct main app and run it'''
    #app, _win = get_main_app(sys.argv)
    app=QtWidgets.QApplication(sys.argv)
    login = LoginDialog()
    login.show()
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main())

#changed by Nie, Qiu
#changed by Junjie Wang
