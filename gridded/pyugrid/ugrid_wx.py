"""
ugid_wx.py:

A small wxPython utility app to visualize pyugrids, etc.
"""


import os
try:
    import wx
except ImportError as err:
    raise ImportError("\n This script needs the wxPython package."
                      "\n You can try installing it with: conda install wxpython"
                      "\n or find it at wxpython.org"
                      )
    raise

from .ugrid import UGrid

# Import the installed version.
from wx.lib.floatcanvas import NavCanvas, FloatCanvas

# FIXME: Need to add a GUI for re-set these at some point...
preferences = {'draw_indexes': True,
               'draw_cells': True,
               'draw_boundaries': True,
               'node_diameter': 4,
               }


class DrawFrame(wx.Frame):

    """
    A frame used for the ugrid viewer.

    """

    # Some parameters for drawing:
    background_color = (200, 200, 200)  # Grey

    label_size = 16
    label_color = 'black'
    label_background_color = background_color

    node_color = 'black'

    face_color = 'cyan'
    face_edge_color = 'black'

    edge_color = 'red'

    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)

        self.CreateStatusBar()

        MenuBar = wx.MenuBar()

        FileMenu = wx.Menu()

        item = FileMenu.Append(wx.ID_EXIT, text="&Exit")
        self.Bind(wx.EVT_MENU, self.OnQuit, item)

        item = FileMenu.Append(wx.ID_ANY, text="&Open")
        self.Bind(wx.EVT_MENU, self.OnOpen, item)

        item = FileMenu.Append(wx.ID_ANY, text="&Save Image")
        self.Bind(wx.EVT_MENU, self.OnSaveImage, item)

        MenuBar.Append(FileMenu, "&File")
        self.SetMenuBar(MenuBar)

        # Add the Canvas
        Canvas = NavCanvas.NavCanvas(self, -1,
                                     size=(500, 500),
                                     ProjectionFun=None,
                                     Debug=0,
                                     BackgroundColor=self.background_color,
                                     ).Canvas

        self.Canvas = Canvas

        FloatCanvas.EVT_MOTION(self.Canvas, self.OnMove)

        self.Show()
        Canvas.ZoomToBB()

    def Draw_UGRID(self, grid):
        """
        Draws a UGRID Object.
        """

        self.Canvas.ClearAll()
        # add the elements:
        nodes = grid.nodes
        # add the elements:
        if grid.faces is not None:
            for i, f in enumerate(grid.faces):
                face = nodes[f]
                self.Canvas.AddPolygon(face, FillColor=self.face_color,
                                       LineColor=self.face_edge_color,
                                       LineWidth=2)
                mid = face.mean(axis=0)
                if preferences['draw_indexes']:
                    self.Canvas.AddText(repr(i), mid, Size=self.label_size,
                                        Position='cc')

        # Add the edges:
        if grid.edges is not None:
            for i, e in enumerate(grid.edges):
                edge = nodes[e]
                self.Canvas.AddLine(edge, LineColor=self.edge_color,
                                    LineWidth=3)
                if preferences['draw_indexes']:
                    mid = edge.mean(axis=0)
                    background_color = self.label_background_color
                    self.Canvas.AddText(repr(i),
                                        mid,
                                        Size=self.label_size,
                                        Position='cc',
                                        Color=self.label_color,
                                        BackgroundColor=background_color)

        # Add the Nodes.
        if preferences['draw_indexes']:
            for i, n in enumerate(nodes):
                background_color = self.label_background_color
                self.Canvas.AddText(repr(i), n, Size=self.label_size,
                                    BackgroundColor=background_color)
        self.Canvas.AddPointSet(nodes, Diameter=preferences['node_diameter'],
                                Color=self.node_color)
        self.Canvas.ZoomToBB()

    def load_ugrid_file(self, filename):
        grid = UGrid.from_ncfile(filename)
        self.Draw_UGRID(grid)

    def OnMove(self, event):
        """
        Updates the status bar with the world coordinates.

        """
        self.SetStatusText("%.2f, %.2f" % tuple(event.Coords))

    def OnQuit(self, Event):
        self.Destroy()

    def OnOpen(self, event):
        dlg = wx.FileDialog(self, 'Choose a ugrid file to open', '.', '',
                            '*.nc', wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()
            filename = os.path.abspath(filename)
            self.load_ugrid_file(filename)
        dlg.Destroy()

    def OnSaveImage(self, event):
        dlg = wx.FileDialog(self, 'Save a PNG file', ".", '', '*.png', wx.SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()
            self.save_image(filename)
        dlg.Destroy()

    def save_image(self, filename):
        self.Canvas.SaveAsImage(filename)


def main():
    import sys
    app = wx.App(False)
    F = DrawFrame(None, title="UGRID Test App", size=(700, 700))

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        F.load_ugrid_file(filename)
    else:
        from pyugrid import test_examples
        F.Draw_UGRID(test_examples.twenty_one_triangles())

    app.MainLoop()


if __name__ == "__main__":
    main()
