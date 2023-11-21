package core ; 
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.Program;

import org.moeaframework.util.tree.*;

/*
 * A Viewer for Binary Trees.
 */
public class TreeView {

    /* The tree currently being display */
    protected Node tree;

    /* The max. height of any tree drawn so far.  This
       is used to avoid the tree jumping around when nodes
       are removed from the tree. */
    protected int maxHeight;

    /* The font for the tree nodes. */
    protected Font font = new Font("Roman", 0, 12);
    BufferedImage bimage ; 
    Graphics2D ig2 ; 
    int width ; 
    int height ; 
    
    /* 
     * Create a new window with the given width and height 
     * that is showing the given tree.
     */
    public TreeView(Node node, int width, int height) {
    	
    	this.bimage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
    	this.ig2 = bimage.createGraphics();
        //Initialize drawing colors, border, opacity.
    	ig2.setBackground(Color.white);
    	ig2.setColor(Color.black);
        
        // Create window and make it so hitting the close icon
        // will terminate the program
       
        this.width = width ; 
        this.height = height ; 
        // install initial tree.
        setTree(node);
    }

    /*
     * Set the display to show the given tree.
     */ 
    public void setTree(Node t) {
        tree = t;
        maxHeight = t.getMaximumHeight() ; 
    }

    /*
     * Invoke this method whenever you would like the window
     * to be refreshed, such as after updating the tree in some
     * way.
     */
  


    /*
     * Draw the contents of the tree into the given Graphics
     * context.
     * It will place all info between minX and maxX in the x-direction,
     * starting at location y in the y-direction.  Levels of the tree
     * will be separated by yStep pixels.
     */
    protected void drawTree(Graphics2D g, int minX, int maxX, 
                            int y, int yStep, Node tree) {

    	
        
   
        String s = tree.getName();
        g.setFont(font);
        FontMetrics fm = g.getFontMetrics();
        int width = fm.stringWidth(s);
        int height = fm.getHeight();

        int xSep = Math.min((maxX - minX)/2, 10);

        g.drawString(s, (minX + maxX)/2 - width/2, y + yStep/2);
        if (tree.getNumberOfArguments() != 0 ) {
            // if left tree not empty, draw line to it and recursively
            // draw that tree
            g.drawLine((minX + maxX)/2 - xSep, y + yStep/2 + 5,
                       (minX + (minX + maxX)/2) / 2, 
                       y + yStep + yStep/2 - height);
            drawTree(g, minX, (minX + maxX)/2, y + yStep, yStep, tree.getArgument(0));
            // same thing for right subtree.
            g.drawLine((minX + maxX)/2 + xSep, y + yStep/2 + 5,
                       (maxX + (minX + maxX)/2) / 2, 
                       y + yStep + yStep/2 - height);
            drawTree(g, (minX + maxX)/2, maxX, y + yStep, yStep,  tree.getArgument(1));
        }
    }
    public void save(String filename)
    {
    	try {
    		this.drawTree(ig2, 0, width, 0, height / (2*tree.getMaximumHeight() + 1),  tree);
			ImageIO.write(this.bimage, "PNG", new File(filename));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }


    /*
     * paint method unherited from the Swing library.  Just
     * calls drawTree whenever the window needs to be repainted.
     */
   /* protected void paintComponent(Graphics g) {
        super.paintComponent(g);      //clears the background
        int width = getWidth();
        int height = getHeight();
        maxHeight = Math.max(tree.getMaximumHeight(),maxHeight);
        int treeHeight = maxHeight;

        drawTree(g, 0, width, 0, height / (treeHeight + 1), tree);

    } */

    /* 
     * Test code.
     
    public static void main(String s[]) {
        BinaryTree<String> tree = new BinaryTree<String>("Hiya", 
                                         new BinaryTree<String>("moo"), 
                                         new BinaryTree<String>("cow"));
        BinaryTree<String> tree2 = new BinaryTree<String>("MOOOOOO", tree, tree);
        BinaryTreeView<String> btv = new BinaryTreeView<String>(tree2, 400, 400);
        btv.refresh();
    }*/
}