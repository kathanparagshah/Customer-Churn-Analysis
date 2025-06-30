import { extendTheme } from '@chakra-ui/react';

const theme = extendTheme({
  colors: {
    primary: {
      50: '#F0F4F8',
      100: '#D9E2EC',
      200: '#BCCCDC',
      300: '#9FB3C8',
      400: '#829AB1',
      500: '#1A365D', // Deep navy
      600: '#153E75',
      700: '#1A365D',
      800: '#153E75',
      900: '#102A43',
    },
    secondary: {
      50: '#F8FAFC',
      100: '#F1F5F9',
      200: '#E2E8F0',
      300: '#CBD5E0',
      400: '#A0AEC0',
      500: '#64748B', // Slate gray
      600: '#475569',
      700: '#334155',
      800: '#1E293B',
      900: '#0F172A',
    },
    accent: {
      50: '#FFFBEB',
      100: '#FEF3C7',
      200: '#FDE68A',
      300: '#FCD34D',
      400: '#FBBF24',
      500: '#D4AF37', // Muted gold
      600: '#B8860B',
      700: '#9A7B0A',
      800: '#7C6309',
      900: '#5E4B07',
    },
    background: {
      50: '#FFFFFF',
      100: '#F8FAFC',
      200: '#F1F5F9',
      300: '#E2E8F0',
    },
  },
  fonts: {
    heading: '"Merriweather", serif',
    body: '"Inter", sans-serif',
  },
  space: {
    base: '24px',
  },
  components: {
    Button: {
      baseStyle: {
        fontWeight: '600',
        borderRadius: 'md',
        transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
      },
      variants: {
        solid: {
          bg: 'primary.500',
          color: 'white',
          _hover: {
            bg: 'primary.600',
            transform: 'translateY(-1px)',
            boxShadow: '0 4px 12px rgba(26, 54, 93, 0.15)',
          },
          _active: {
            transform: 'translateY(0)',
          },
        },
        outline: {
          borderColor: 'accent.500',
          color: 'accent.600',
          _hover: {
            bg: 'accent.50',
            borderColor: 'accent.600',
            transform: 'translateY(-1px)',
          },
        },
        ghost: {
          color: 'secondary.600',
          _hover: {
            bg: 'secondary.50',
            color: 'secondary.700',
          },
        },
        cta: {
          bg: 'linear-gradient(135deg, #D4AF37 0%, #B8860B 100%)',
          color: 'white',
          fontWeight: '700',
          px: 8,
          py: 3,
          fontSize: 'lg',
          _hover: {
            transform: 'translateY(-2px)',
            boxShadow: '0 8px 25px rgba(212, 175, 55, 0.3)',
          },
        },
      },
    },
    Card: {
      baseStyle: {
        container: {
          borderRadius: 'lg',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          border: 'none',
          _hover: {
            transform: 'translateY(-4px)',
            boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1), 0 4px 10px rgba(0, 0, 0, 0.05)',
          },
        },
      },
    },
    Input: {
      variants: {
        filled: {
          field: {
            bg: 'background.100',
            borderColor: 'transparent',
            borderRadius: 'md',
            _hover: {
              bg: 'background.200',
            },
            _focus: {
              bg: 'white',
              borderColor: 'accent.500',
              boxShadow: '0 0 0 1px var(--chakra-colors-accent-500)',
            },
          },
        },
      },
      defaultProps: {
        variant: 'filled',
      },
    },
    Table: {
      variants: {
        simple: {
          th: {
            borderColor: 'secondary.200',
            color: 'secondary.600',
            fontSize: 'sm',
            fontWeight: '600',
            textTransform: 'uppercase',
            letterSpacing: 'wider',
          },
          td: {
            borderColor: 'secondary.100',
          },
        },
      },
    },
    Tabs: {
      variants: {
        line: {
          tablist: {
            borderColor: 'secondary.200',
          },
          tab: {
            color: 'secondary.600',
            _selected: {
              color: 'primary.500',
              borderColor: 'accent.500',
            },
          },
        },
      },
    },
    Stat: {
      baseStyle: {
        container: {
          bg: 'white',
          p: 6,
          borderRadius: 'lg',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          _hover: {
            transform: 'translateY(-2px)',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
          },
        },
      },
    },
  },
  styles: {
    global: {
      body: {
        bg: 'background.100',
        color: 'secondary.700',
        fontFamily: 'body',
      },
      '*': {
        borderColor: 'secondary.200',
      },
    },
  },
});

export default theme;