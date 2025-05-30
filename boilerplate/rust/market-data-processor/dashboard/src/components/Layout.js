import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  AppBar, 
  Box, 
  Toolbar, 
  Typography, 
  Drawer, 
  List, 
  ListItem, 
  ListItemButton, 
  ListItemIcon, 
  ListItemText, 
  ListSubheader,
  Collapse,
  Divider, 
  IconButton, 
  Badge, 
  Chip,
  Tooltip,
  Breadcrumbs,
  Link,
  useMediaQuery,
  useTheme
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  History as HistoryIcon,
  Settings as SettingsIcon,
  Notifications as NotificationsIcon,
  TrendingUp as TrendingUpIcon,
  BarChart as BarChartIcon,
  ShowChart as ShowChartIcon,
  Timeline as TimelineIcon,
  AccountBalance as AccountBalanceIcon,
  Tune as TuneIcon,
  MonitorHeart as MonitorHeartIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
  ExpandLess as ExpandLessIcon,
  ExpandMore as ExpandMoreIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Home as HomeIcon,
  KeyboardArrowRight as KeyboardArrowRightIcon
} from '@mui/icons-material';

const drawerWidth = 240;
const collapsedDrawerWidth = 72;

function Layout({ children }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [drawerCollapsed, setDrawerCollapsed] = useState(false);
  const [expandedSection, setExpandedSection] = useState('');
  const [notificationCount, setNotificationCount] = useState(2);
  const [systemStatus, setSystemStatus] = useState('OK'); // 'OK', 'WARNING', 'ERROR'
  
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  useEffect(() => {
    // Set up keyboard shortcuts
    const handleKeyDown = (event) => {
      // Alt + key shortcuts
      if (event.altKey) {
        switch (event.key) {
          case 'o': // Alt+O for Overview
            navigate('/');
            break;
          case 't': // Alt+T for Trading
            navigate('/trading');
            break;
          case 'a': // Alt+A for Analytics
            navigate('/analytics');
            break;
          case 'm': // Alt+M for Market Data
            navigate('/market-data');
            break;
          case 's': // Alt+S for Settings
            navigate('/settings');
            break;
          case 'r': // Alt+R for Monitoring
            navigate('/monitoring');
            break;
          default:
            break;
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [navigate]);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };
  
  const toggleDrawerCollapse = () => {
    setDrawerCollapsed(!drawerCollapsed);
  };
  
  const handleSectionClick = (section) => {
    setExpandedSection(expandedSection === section ? '' : section);
  };
  
  // Navigation sections with subsections for organization
  const navigationSections = [
    {
      id: 'overview',
      title: 'Overview',
      icon: <DashboardIcon />,
      path: '/',
      shortcut: 'Alt+O',
      expandable: false
    },
    {
      id: 'trading',
      title: 'Trading',
      icon: <TrendingUpIcon />,
      path: '/trading',
      shortcut: 'Alt+T',
      expandable: false
    },
    {
      id: 'analytics',
      title: 'Analytics',
      icon: <BarChartIcon />,
      path: '/analytics',
      shortcut: 'Alt+A',
      expandable: false
    },
    {
      id: 'market-data',
      title: 'Market Data',
      icon: <ShowChartIcon />,
      path: '/market-data',
      shortcut: 'Alt+M',
      expandable: false
    },
    {
      id: 'history',
      title: 'History',
      icon: <HistoryIcon />,
      expandable: true,
      items: [
        { title: 'Order History', path: '/order-history' },
        { title: 'Trade History', path: '/trade-history' }
      ]
    },
    {
      id: 'settings',
      title: 'Settings',
      icon: <SettingsIcon />,
      path: '/settings',
      shortcut: 'Alt+S',
      expandable: false
    },
    {
      id: 'monitoring',
      title: 'Monitoring',
      icon: <MonitorHeartIcon />,
      path: '/monitoring',
      shortcut: 'Alt+R',
      expandable: false
    }
  ];
  
  // Generate breadcrumbs based on current path
  const generateBreadcrumbs = () => {
    const pathnames = location.pathname.split('/').filter(x => x);
    
    if (pathnames.length === 0) {
      return [
        <Typography key="dashboard" color="text.primary">Dashboard</Typography>
      ];
    }
    
    return [
      <Link
        underline="hover"
        key="home"
        color="inherit"
        href="/"
        onClick={(e) => {
          e.preventDefault();
          navigate('/');
        }}
      >
        Dashboard
      </Link>,
      ...pathnames.map((value, index) => {
        const last = index === pathnames.length - 1;
        const to = `/${pathnames.slice(0, index + 1).join('/')}`;
        
        const formattedValue = value
          .split('-')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ');
          
        return last ? (
          <Typography key={to} color="text.primary">
            {formattedValue}
          </Typography>
        ) : (
          <Link
            underline="hover"
            key={to}
            color="inherit"
            href={to}
            onClick={(e) => {
              e.preventDefault();
              navigate(to);
            }}
          >
            {formattedValue}
          </Link>
        );
      })
    ];
  };

  const drawer = (
    <div>
      <Toolbar sx={{ justifyContent: 'space-between', height: 64 }}>
        {!drawerCollapsed ? (
          <>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Typography variant="h6" noWrap component="div">
                Trading Agent
              </Typography>
              <Chip 
                label="Paper Trading" 
                color="primary" 
                size="small" 
                sx={{ ml: 1 }}
              />
            </Box>
            <IconButton onClick={toggleDrawerCollapse} edge="end">
              <ChevronLeftIcon />
            </IconButton>
          </>
        ) : (
          <IconButton onClick={toggleDrawerCollapse} sx={{ mx: 'auto' }}>
            <ChevronRightIcon />
          </IconButton>
        )}
      </Toolbar>
      <Divider />
      
      {/* System status indicator */}
      <Box 
        sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          p: 1, 
          bgcolor: systemStatus === 'OK' ? 'success.dark' : 
                  systemStatus === 'WARNING' ? 'warning.dark' : 'error.dark',
          color: 'white',
          justifyContent: drawerCollapsed ? 'center' : 'flex-start'
        }}
      >
        {systemStatus === 'OK' ? <CheckCircleIcon fontSize="small" /> : 
         systemStatus === 'WARNING' ? <WarningIcon fontSize="small" /> : 
         <ErrorIcon fontSize="small" />}
         
        {!drawerCollapsed && (
          <Typography variant="body2" sx={{ ml: 1 }}>
            System: {systemStatus}
          </Typography>
        )}
      </Box>
      
      {/* Navigation sections */}
      {navigationSections.map(section => (
        <React.Fragment key={section.id}>
          {section.expandable ? (
            <>
              <ListItem disablePadding>
                <ListItemButton 
                  onClick={() => handleSectionClick(section.id)}
                  sx={{
                    minHeight: 48,
                    justifyContent: drawerCollapsed ? 'center' : 'initial',
                    px: 2.5,
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: 0,
                      mr: drawerCollapsed ? 'auto' : 3,
                      justifyContent: 'center',
                    }}
                  >
                    <Tooltip title={drawerCollapsed ? section.title : ''}>
                      {section.icon}
                    </Tooltip>
                  </ListItemIcon>
                  {!drawerCollapsed && (
                    <>
                      <ListItemText primary={section.title} />
                      {expandedSection === section.id ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </>
                  )}
                </ListItemButton>
              </ListItem>
              
              {!drawerCollapsed && (
                <Collapse in={expandedSection === section.id} timeout="auto" unmountOnExit>
                  <List component="div" disablePadding>
                    {section.items.map(item => (
                      <ListItem key={item.title} disablePadding>
                        <ListItemButton 
                          selected={location.pathname === item.path}
                          onClick={() => {
                            navigate(item.path);
                            setMobileOpen(false);
                          }}
                          sx={{ pl: 4 }}
                        >
                          <ListItemIcon>
                            <KeyboardArrowRightIcon />
                          </ListItemIcon>
                          <ListItemText primary={item.title} />
                        </ListItemButton>
                      </ListItem>
                    ))}
                  </List>
                </Collapse>
              )}
            </>
          ) : (
            <ListItem disablePadding>
              <ListItemButton 
                selected={location.pathname === section.path}
                onClick={() => {
                  navigate(section.path);
                  setMobileOpen(false);
                }}
                sx={{
                  minHeight: 48,
                  justifyContent: drawerCollapsed ? 'center' : 'initial',
                  px: 2.5,
                }}
              >
                <Tooltip title={drawerCollapsed ? `${section.title} (${section.shortcut})` : ''}>
                  <ListItemIcon
                    sx={{
                      minWidth: 0,
                      mr: drawerCollapsed ? 'auto' : 3,
                      justifyContent: 'center',
                      color: location.pathname === section.path ? 'primary.main' : 'inherit'
                    }}
                  >
                    {section.icon}
                  </ListItemIcon>
                </Tooltip>
                {!drawerCollapsed && (
                  <>
                    <ListItemText primary={section.title} />
                    <Typography variant="caption" color="text.secondary">
                      {section.shortcut}
                    </Typography>
                  </>
                )}
              </ListItemButton>
            </ListItem>
          )}
        </React.Fragment>
      ))}
      
      <Divider sx={{ mt: 'auto' }} />
      <Box sx={{ p: drawerCollapsed ? 1 : 2, textAlign: drawerCollapsed ? 'center' : 'left' }}>
        {!drawerCollapsed && (
          <>
            <Typography variant="body2" color="text.secondary">
              Market Data Processor v1.0.0
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Running in paper trading mode
            </Typography>
          </>
        )}
      </Box>
    </div>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerCollapsed ? collapsedDrawerWidth : drawerWidth}px)` },
          ml: { sm: `${drawerCollapsed ? collapsedDrawerWidth : drawerWidth}px` },
          transition: theme => theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          
          {/* Breadcrumbs navigation */}
          <Breadcrumbs 
            aria-label="breadcrumb" 
            separator={<KeyboardArrowRightIcon fontSize="small" />}
            sx={{ flexGrow: 1, color: 'white' }}
          >
            {generateBreadcrumbs()}
          </Breadcrumbs>
          
          {/* System status indicator */}
          <Chip 
            icon={systemStatus === 'OK' ? <CheckCircleIcon /> : <ErrorIcon />}
            label={systemStatus} 
            color={systemStatus === 'OK' ? 'success' : 'error'} 
            variant="outlined"
            sx={{ mr: 2, borderColor: 'white', color: 'white' }}
            onClick={() => navigate('/monitoring')}
          />
          
          {/* Notifications */}
          <Tooltip title="Notifications">
            <IconButton color="inherit">
              <Badge badgeContent={notificationCount} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>
      
      {/* Navigation drawer */}
      <Box
        component="nav"
        sx={{ 
          width: { sm: drawerCollapsed ? collapsedDrawerWidth : drawerWidth }, 
          flexShrink: { sm: 0 },
          transition: theme => theme.transitions.create('width', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
          }),
        }}
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth 
            },
          }}
        >
          {drawer}
        </Drawer>
        
        {/* Desktop drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerCollapsed ? collapsedDrawerWidth : drawerWidth,
              transition: theme => theme.transitions.create('width', {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.enteringScreen,
              }),
              overflowX: 'hidden',
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      
      {/* Main content */}
      <Box
        component="main"
        sx={{ 
          flexGrow: 1, 
          p: 3, 
          width: { sm: `calc(100% - ${drawerCollapsed ? collapsedDrawerWidth : drawerWidth}px)` },
          marginTop: '64px',
          transition: theme => theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
          }),
        }}
      >
        {children}
      </Box>
    </Box>
  );
}

export default Layout;
