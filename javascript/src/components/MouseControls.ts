import {
	EventDispatcher,
	Matrix4,
	Plane,
	Raycaster,
	Vector2,
	Vector3,
  Object3D,
  Camera
} from "three";

class MouseControls extends EventDispatcher {
  _camera: Camera;
  _domElement: HTMLElement;

  _objects: Object3D[] = [];
  _keys: number[] = [];
	_plane: Plane = new Plane();
	_raycaster: Raycaster = new Raycaster();

	_mouse: Vector2 = new Vector2();
	_offset: Vector3 = new Vector3();
	_intersection: Vector3 = new Vector3();
	_worldPosition: Vector3 = new Vector3();
	_inverseMatrix: Matrix4 = new Matrix4();
	_intersections = [];

  _selected: Object3D = null;
  _hovered: Object3D = null;


	enabled: boolean = true;
	transformGroup: boolean = false;
  
  _dragHandler: (obj: Object3D, pos: Vector3) => void = null;

	constructor(_camera: Camera, _domElement: HTMLElement) {
    super();

    this._camera = _camera;
    this._domElement = _domElement;

		this._domElement.addEventListener( 'pointermove', this.onDocumentMouseMove, false );
		this._domElement.addEventListener( 'pointerdown', this.onDocumentMouseDown, true );
		this._domElement.addEventListener( 'pointerup', this.onDocumentMouseCancel, false );
		this._domElement.addEventListener( 'mouseleave', this.onDocumentMouseCancel, false );
		this._domElement.addEventListener( 'touchmove', this.onDocumentTouchMove, false );
		this._domElement.addEventListener( 'touchstart', this.onDocumentTouchStart, false );
		this._domElement.addEventListener( 'touchend', this.onDocumentTouchEnd, false );

	}

	deactivate = () => {
		this._domElement.removeEventListener( 'pointermove', this.onDocumentMouseMove, false );
		this._domElement.removeEventListener( 'pointerdown', this.onDocumentMouseDown, false );
		this._domElement.removeEventListener( 'pointerup', this.onDocumentMouseCancel, false );
		this._domElement.removeEventListener( 'mouseleave', this.onDocumentMouseCancel, false );
		this._domElement.removeEventListener( 'touchmove', this.onDocumentTouchMove, false );
		this._domElement.removeEventListener( 'touchstart', this.onDocumentTouchStart, false );
		this._domElement.removeEventListener( 'touchend', this.onDocumentTouchEnd, false );

		this._domElement.style.cursor = '';
	}

	dispose = () => {
		this.deactivate();
	}

	getObjects = () => {
		return this._objects;
  }

  setInteractiveList = (objects: Object3D[], keys: number[]) => {
    this._objects = objects;
    this._keys = keys;
  }
  
  /*
  add = (object: Object3D) => {
    const index = this._objects.indexOf(object);
    if (index === -1)
      this._objects.push(object);
    console.log("Adding object, now have "+this._objects.length);
  }
  
  remove = (object: Object3D) => {
    console.log("Testing to remove object");
    const index = this._objects.indexOf(object);
    if (index > -1) {
      console.log("Removing object of "+this._objects.length);
      this._objects.splice(index, 1);
    }
  }
  */

	onDocumentMouseMove = (event: MouseEvent) => {
		event.preventDefault();

		const rect = this._domElement.getBoundingClientRect();

		this._mouse.x = ( ( event.clientX - rect.left ) / rect.width ) * 2 - 1;
		this._mouse.y = - ( ( event.clientY - rect.top ) / rect.height ) * 2 + 1;

		this._raycaster.setFromCamera( this._mouse, this._camera );

		if ( this._selected && this.enabled ) {

			if ( this._raycaster.ray.intersectPlane( this._plane, this._intersection ) ) {

        this._dragHandler(this._selected, this._intersection.sub( this._offset ).applyMatrix4( this._inverseMatrix ));
				// _selected.position.copy( _intersection.sub( _offset ).applyMatrix4( _inverseMatrix ) );

			}

			this.dispatchEvent( { type: 'drag', object: this._selected } );

			return;

		}

		this._intersections.length = 0;

		this._raycaster.setFromCamera(this._mouse, this._camera );
		this._raycaster.intersectObjects(this._objects, true, this._intersections);

		if (this._intersections.length > 0) {

			var object = this._intersections[ 0 ].object;

			this._plane.setFromNormalAndCoplanarPoint( this._camera.getWorldDirection( this._plane.normal ), this._worldPosition.setFromMatrixPosition( object.matrixWorld ) );

			if ( this._hovered !== object ) {

        const key = this._keys[this._objects.indexOf(object)];

				this.dispatchEvent( { type: 'hoveron', object: object, key: key, top_x: event.clientX - rect.left, top_y: event.clientY - rect.top } );

				this._domElement.style.cursor = 'pointer';
				this._hovered = object;

			}

		} else {

			if ( this._hovered !== null ) {

        const key = this._keys[this._objects.indexOf(object)];

				this.dispatchEvent( { type: 'hoveroff', object: this._hovered, key: key, top_x: this._mouse.x, top_y: this._mouse.y } );

				this._domElement.style.cursor = 'auto';
				this._hovered = null;

			}

		}

	}

	onDocumentMouseDown = (event: MouseEvent) => {

		event.preventDefault();

		this._intersections.length = 0;

		this._raycaster.setFromCamera(this._mouse, this._camera);
		this._raycaster.intersectObjects(this._objects, true, this._intersections);

		if (this._intersections.length > 0) {
			this._selected = ( this.transformGroup === true ) ? this._objects[ 0 ] : this._intersections[ 0 ].object;

			if ( this._raycaster.ray.intersectPlane( this._plane, this._intersection ) ) {

				this._inverseMatrix.getInverse( this._selected.parent.matrixWorld );
				this._offset.copy( this._intersection ).sub( this._worldPosition.setFromMatrixPosition( this._selected.matrixWorld ) );

			}

			this._domElement.style.cursor = 'move';

			this.dispatchEvent( { type: 'dragstart', object: this._selected } );

		}


	}

	onDocumentMouseCancel = (event: MouseEvent) => {
		event.preventDefault();

		if ( this._selected ) {
			this.dispatchEvent( { type: 'dragend', object: this._selected } );

			this._selected = null;
		}

		this._domElement.style.cursor = this._hovered ? 'pointer' : 'auto';
  }
  
  setDragHandler = (dragHandler: (obj: Object3D, pos: Vector3) => void) => {
    this._dragHandler = dragHandler;
  }

	onDocumentTouchMove = (event: TouchEvent) => {

		event.preventDefault();
		const touchList = event.changedTouches[ 0 ];

		var rect = this._domElement.getBoundingClientRect();

		this._mouse.x = ( ( touchList.clientX - rect.left ) / rect.width ) * 2 - 1;
		this._mouse.y = - ( ( touchList.clientY - rect.top ) / rect.height ) * 2 + 1;

		this._raycaster.setFromCamera(this._mouse, this._camera);

		if ( this._selected && this.enabled ) {

			if ( this._raycaster.ray.intersectPlane( this._plane, this._intersection ) ) {

        this._dragHandler(this._selected, this._intersection.sub( this._offset ).applyMatrix4( this._inverseMatrix ));
				// _selected.position.copy( _intersection.sub( _offset ).applyMatrix4( _inverseMatrix ) );
			}

			this.dispatchEvent( { type: 'drag', object: this._selected } );

			return;

		}

	}

	onDocumentTouchStart = (event : TouchEvent) => {

		event.preventDefault();
		const touchList = event.changedTouches[ 0 ];

		var rect = this._domElement.getBoundingClientRect();

		this._mouse.x = ( ( touchList.clientX - rect.left ) / rect.width ) * 2 - 1;
		this._mouse.y = - ( ( touchList.clientY - rect.top ) / rect.height ) * 2 + 1;

		this._intersections.length = 0;

		this._raycaster.setFromCamera(this._mouse, this._camera );
		this._raycaster.intersectObjects(this._objects, true, this._intersections );

		if (this._intersections.length > 0) {

			this._selected = ( this.transformGroup === true ) ? this._objects[ 0 ] : this._intersections[ 0 ].object;

			this._plane.setFromNormalAndCoplanarPoint( this._camera.getWorldDirection( this._plane.normal ), this._worldPosition.setFromMatrixPosition( this._selected.matrixWorld ) );

			if ( this._raycaster.ray.intersectPlane( this._plane, this._intersection ) ) {

				this._inverseMatrix.getInverse( this._selected.parent.matrixWorld );
				this._offset.copy( this._intersection ).sub( this._worldPosition.setFromMatrixPosition( this._selected.matrixWorld ) );

			}

			this._domElement.style.cursor = 'move';

			this.dispatchEvent( { type: 'dragstart', object: this._selected } );

		}


	}

	onDocumentTouchEnd = ( event: TouchEvent ) => {

		event.preventDefault();

		if ( this._selected ) {

			this.dispatchEvent( { type: 'dragend', object: this._selected } );

			this._selected = null;

		}

		this._domElement.style.cursor = 'auto';

	}
};

export default MouseControls;
