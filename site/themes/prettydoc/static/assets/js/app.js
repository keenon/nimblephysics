(function ($) {
  "use strict";

  var fn = {

    // Launch Functions
    Launch: function () {
      fn.Header();
      fn.Masonry();
      fn.Overlay();
      fn.Filetree();
      fn.Clipboard();
      fn.OwlCarousel();
      fn.ImageView();
      fn.Apps();
    },


    Header: function (){
      $(document.body).headroom({
        tolerance : 10
      });
    },


    Masonry: function() {
      var $grid = $('.masonry').masonry({
        itemSelector: '.masonry > *',
      });
    },


    // Owl Carousel
    OwlCarousel: function() {

      $('.owl-carousel').each(function() {
        var a = $(this),
          items = a.data('items') || [1,1,1,1],
          margin = a.data('margin'),
          loop = a.data('loop'),
          nav = a.data('nav'),
          dots = a.data('dots'),
          center = a.data('center'),
          autoplay = a.data('autoplay'),
          autoplaySpeed = a.data('autoplay-speed'),
          rtl = a.data('rtl'),
          autoheight = a.data('autoheight');

        var options = {
          nav: nav || false,
          loop: loop || false,
          dots: dots || false,
          center: center || false,
          autoplay: autoplay || false,
          autoHeight: autoheight || false,
          rtl: rtl || false,
          margin: margin || 0,
          autoplayTimeout: autoplaySpeed || 3000,
          autoplaySpeed: 400,
          autoplayHoverPause: true,
          responsive: {
            0: { items: items[3] || 1 },
            992: { items: items[2] || 1 },
            1200: { items: items[1] || 1 },
            1600: { items: items[0] || 1}
          }
        };

        a.owlCarousel(options);

        // Custom Navigation Events
        $(document).on('click', '.owl-item>div', function() {
          $owl.trigger('to.owl.carousel', $(this).data( 'position' ) );
        });
      });
    },

    // Overlay Menu
    Overlay: function() {
      $(document).ready(function(){
        $('.overlay-menu-open').click(function(){
          $(this).toggleClass('active');
          $('html').toggleClass('active');
          $('.overlay-menu').toggleClass('active');
        });
      });
    },


    // File Tree
    Filetree: function() {
      var folder = $('.file-tree li.file-tree-folder'),
          file = $('.file-tree li');

      folder.on("click", function(a) {
          $(this).children('ul').slideToggle(400, function() {
              $(this).parent("li").toggleClass("open")
          }), a.stopPropagation()
      })

      file.on('click', function(b){
        b.stopPropagation();
      })
    },


    // Clipboard
    Clipboard: function() {
      var a = new ClipboardJS('.anchor', {
        text: function(b) {
          return window.location.host + window.location.pathname + $(b).attr("href")
        }
      });

      a.on('success', function(e) {
        e.clearSelection(), $(e.trigger).addClass("copied"), setTimeout(function() {
          $(e.trigger).removeClass("copied")
        }, 2000)
      });
    },

    ImageView: function() {
      $('.lightbox').magnificPopup({
        type: 'image',
        closeOnContentClick: true,
        closeBtnInside: false,
        fixedContentPos: true,
        mainClass: 'mfp-no-margins mfp-with-zoom', // class to remove default margin from left and right side
        image: {
          verticalFit: true
        }
      });

      $('.gallery').each(function() { // the containers for all your galleries
          $(this).magnificPopup({
              delegate: '.photo > a', // the selector for gallery item
              type: 'image',
              mainClass: 'mfp-no-margins mfp-with-zoom', // class to remove default margin from left and right side
              gallery: {
                enabled:true
              }
          });
      });
      
      $('.popup-youtube, .popup-vimeo, .popup-gmaps').magnificPopup({
        disableOn: 700,
        type: 'iframe',
        mainClass: 'mfp-fade',
        removalDelay: 160,
        preloader: false,

        fixedContentPos: false
      });
    },


    // Apps
    Apps: function () {

      // accordion
      $(document).ready(function() {

        $('.collapse').on('show.bs.collapse', function () {
            $(this).parent().addClass('active');
        });

        $('.collapse').on('hide.bs.collapse', function () {
            $(this).parent().removeClass('active');
        });

      });


      // tooltips
      $('[data-toggle="tooltip"]').tooltip()



      // skrollr
      skrollr.init({  
          forceHeight: false,        
          mobileCheck: function() {
              //hack - forces mobile version to be off
              return false;
          }
      });


      // Smooth Scroll
      $(function () {
        var scroll = new SmoothScroll('[data-scroll]');
      });


      // Lavalamp
      $('.lavalamp').lavalamp({
        setOnClick: true,
        enableHover: false,
        margins: false,
        autoUpdate: true,
        duration: 200
      });


      $(document).ready(function(){
          var window_width = jQuery( window ).width();

          if (window_width < 768) {
            $(".sticky").trigger("sticky_kit:detach");
          } else {
            make_sticky();
          }


          $( window ).resize(function() {

            window_width = jQuery( window ).width();

            if (window_width < 768) {
              $(".sticky").trigger("sticky_kit:detach");
            } else {
              make_sticky();
            }

          });


          // recalc on collapse
          $('.nav-item .collapse').on('shown.bs.collapse hidden.bs.collapse', function () {
            $(".sticky").trigger("sticky_kit:recalc");
          });

          function make_sticky() {
            $(".sticky").stick_in_parent();
          }

      });


      // prism
      (function(){
        if (typeof self === 'undefined' || !self.Prism || !self.document) {
          return;
        }

        if (!Prism.plugins.toolbar) {
          console.warn('Copy to Clipboard plugin loaded before Toolbar plugin.');

          return;
        }

        var ClipboardJS = window.ClipboardJS || undefined;

        if (!ClipboardJS && typeof require === 'function') {
          ClipboardJS = require('clipboard');
        }

        var callbacks = [];

        if (!ClipboardJS) {
          var script = document.createElement('script');
          var head = document.querySelector('head');

          script.onload = function() {
            ClipboardJS = window.ClipboardJS;

            if (ClipboardJS) {
              while (callbacks.length) {
                callbacks.pop()();
              }
            }
          };

          script.src = 'https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js';
          head.appendChild(script);
        }

        Prism.plugins.toolbar.registerButton('copy-to-clipboard', function (env) {
          var linkCopy = document.createElement('a');
          linkCopy.textContent = 'Copy';

          if (!ClipboardJS) {
            callbacks.push(registerClipboard);
          } else {
            registerClipboard();
          }

          return linkCopy;

          function registerClipboard() {
            var clip = new ClipboardJS(linkCopy, {
              'text': function () {
                return env.code;
              }
            });

            clip.on('success', function() {
              linkCopy.textContent = 'Copied!';

              resetText();
            });
            clip.on('error', function () {
              linkCopy.textContent = 'Press Ctrl+C to copy';

              resetText();
            });
          }

          function resetText() {
            setTimeout(function () {
              linkCopy.textContent = 'Copy';
            }, 5000);
          }
        });
      })();
    }
  };

  $(document).ready(function () {
    fn.Launch();
  });

})(jQuery);